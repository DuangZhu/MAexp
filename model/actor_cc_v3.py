from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
import copy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
import torchvision.models as models
from einops import rearrange
from model.util import init
torch, nn = try_import_torch()
from model.transformer import Transformer
from model.invariant import SingleAgentEncoder
import math
from einops.layers.torch import Rearrange
import numpy as np
# from timm.models.vision_transformer import PatchEmbed

class actor_cc(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")
        self.q_flag = False
        # state scale
        self.grid_size = 8
        self.map_real_w = 125
        self.map_real_h = 125
        self.max_speed = 8
        self.max_theta = math.pi/3
        self.state_scale = torch.tensor([self.map_real_w, self.map_real_h, self.max_speed, self.max_theta])
        # feature_encoder
        self.encoder = self._build_cnn_model()
        self.transformer = self._build_transformer_model()
        layers = [SingleAgentEncoder(num_grids=64, input_dim=32+16, use_id_embedding = True, use_pos_embedding = True,use_add_id_embedding = False)]
        # layers = [SingleAgentEncoder(num_grids=64, input_dim=128+16, use_id_embedding = True, use_pos_embedding = True,use_add_id_embedding = False)]
        layers += [nn.Linear(128, 2), Rearrange('b (h w) c -> b c h w', h = 8)]
        self.agent_encoder = nn.Sequential(*layers)
        self.to_region = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.to_point = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )
        self.actors = [self.encoder, self.agent_encoder, self.to_region, self.to_point, self.transformer]
        self.critic = []
        self.actor_initialized_parameters = self.actor_parameters()
        
        
    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        # backbone提取特征
        B = input_dict.count
        n = input_dict["obs"]["IDs"].shape[1]
        per_obs = input_dict["obs"]["obs"]
        state_ = input_dict["obs"]["state_"]
        state_ = state_ / self.state_scale.to(state_.device)
        other_obs = input_dict["obs"]["others_obs"]
        ids = input_dict["obs"]["IDs"]
        input = torch.cat((per_obs.unsqueeze(1), other_obs), dim=1).reshape(-1, *per_obs.shape[1:])
        self.features = self.encoder(input)
        actor = self.features.reshape((B,-1,*self.features.shape[-3:]))[:,0]
        others = self.features.reshape((B,-1,*self.features.shape[-3:]))[:,1:]
        actor = rearrange(actor, 'B c h w -> B (h w) c') # 32,64,32
        others = rearrange(others, 'B n c h w -> B n (h w) c') # 32,2,64,32
        
        # for grid map
        grid_map = input_dict["obs"]["grid_map"]
        indices = [2*i+j+k for i in range(0,n) for j in range(0,2*n+1,2*n) for k in range(2)] 
        grid_map = grid_map[:, indices, :, :].view(-1, n, 4, 8, 8)
        grid_map = rearrange(grid_map, 'b n c h w -> (b n) c h w', c = 4, h = self.grid_size, w = self.grid_size)
        grid_feature = self.transformer(rearrange(grid_map, 'b c h w -> b (h w) c'))
        grid_feature = rearrange(grid_feature, '(b n) h c -> b n h c', n = n)
        actor = torch.cat([actor, grid_feature[:,0]], dim = -1)
        others = torch.cat([others, grid_feature[:,1:]], dim = -1)
        # attention
        out = self.agent_encoder[0]([actor,others], ids, state_)
        out = self.agent_encoder[1:](out)
        # output distribution
        out = rearrange(out, 'b c h w -> b c (h w)', c = 2, h = self.grid_size, w = self.grid_size)
        out_region = self.to_region(out[:,0])
        out_point = self.to_point(out[:,1])
        out = torch.cat((out_region, out_point), dim =1)
        return out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType: 
        assert self.features is not None, "must call forward() first"
        B = self.features.shape[0]
        B = B/self.n_agents
        return torch.zeros(int(B), device =self.features.device)

    def actor_parameters(self): 
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self): 
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.critic))
        # return list(self.vf_branch.parameters())
    
    def _build_cnn_model(self, cnn_layers_params = None, use_orthogonal = True, activation_id = 1):
        
        if cnn_layers_params is None:
            cnn_layers_params = [(32, 3, 1, 2), (64, 3, 1, 2), (128, 3, 1, 1), (64, 3, 1, 1), (32, 3, 1, 1)]
            # cnn_layers_params = [(32, 3, 1, 2), (64, 3, 1, 2), (128, 3, 1, 1), (256, 3, 1, 1), (128, 3, 1, 1)]
        else:
            def _convert(params):
                output = []
                for l in params.split(' '):
                    output.append(tuple(map(int, l.split(','))))
                return output
            cnn_layers_params = _convert(cnn_layers_params)
        
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        n_cnn_input = 4
        cnn_layers = []
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if i != len(cnn_layers_params) - 1:
                cnn_layers.append(nn.MaxPool2d(2))

            if i == 0:
                in_channels = n_cnn_input
            else:
                in_channels = self.prev_out_channels

            cnn_layers.append(init_(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
            cnn_layers.append(active_func)
            self.prev_out_channels = out_channels
        cnn_dims = np.array([125,125])
        for i, (_, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if i != len(cnn_layers_params) - 1:
                cnn_dims = self._maxpool_output_dim(dimension=cnn_dims,
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([2, 2], dtype=np.float32),
                stride=np.array([2, 2], dtype=np.float32))
            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )
        
        self.cnn_dims = cnn_dims

        return nn.Sequential(*cnn_layers)

    def _build_transformer_model(self, use_orthogonal = True, activation_id = 1):
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)


        channels = 4
        
        cnn_layers = [init_(nn.Linear(channels,16)),
                    active_func,
                    nn.LayerNorm(16)
                ]
        return nn.Sequential(*cnn_layers)

    def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)   
    
    def _maxpool_output_dim(self, dimension, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)
