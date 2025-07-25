from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
from copy import deepcopy
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
from timm.models.vision_transformer import PatchEmbed, Block, Cross_Attention, CA_Block
from functools import partial
from env_utils.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_encoder
import copy



class Crossat_actor_il(TorchModelV2, nn.Module):

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
        # self.activation = model_config.get("fcnet_activation")
        self.q_flag = False
        # state scale
        self.grid_size = 8
        self.map_real_w = 125
        self.map_real_h = 125
        self.max_speed = 8
        self.max_theta = math.pi/3
        self.state_scale = torch.tensor([self.map_real_w, self.map_real_h, self.max_theta])
        self.mean = torch.Tensor([0.0474, 0.171, 0.0007])
        self.std = torch.Tensor([0.4430, 0.3323, 0.7])
        # model
        self.embed_dim = 128
        drop_rate = 0.
        depth = 6
        norm_layer=nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=128, patch_size=16, in_chans=3, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.state_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.action_token = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, self.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-2)])
        self.critic_blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-2)])
        self.ca_blocks = nn.ModuleList([
            CA_Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])
        self.norm = norm_layer(self.embed_dim)
        self.critic_norm = norm_layer(self.embed_dim)
        self.to_local_region = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.to_local_point  = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        self.others_norm = norm_layer(self.embed_dim)
        self.to_local_state = nn.Linear(self.embed_dim, 2) # 除了baseline2,self.map_real_w+self.map_real_h，其他是2
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=False)
        self.mask_conv.weight.data.fill_(1.0)
        self.to_value = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            # nn.LayerNorm(256, eps=1e-5,elementwise_affine=True),
            nn.Linear(64, 1))
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        # for training
        self.initialize_weights()
        self.actors = [self.patch_embed, self.state_token, self.action_token,
                       self.blocks, self.ca_blocks, self.norm, self.others_norm,
                       self.to_local_point, self.to_local_region, self.to_local_state]
        self.critic = [self.critic_blocks, ]
        self.actor_initialized_parameters = self.actor_parameters()
        
        # # load pretrain
        # state_dict_before = copy.deepcopy(self.state_dict())
        # # put your path of checkpoint here.
        # pretrained_weights = torch.load('/remote-home/ums_zhushaohao/new/2025/TaskExp/results/20250208_ours_randoms_merge1/checkpoint-0.pth')# ---all loss

        # matching_keys = [k for k in pretrained_weights['model'] if k in state_dict_before and pretrained_weights['model'][k].size() == state_dict_before[k].size()]
        # self.load_state_dict(pretrained_weights['model'], strict=False)
        # # # 获取加载权重之后的状态字典
        # state_dict_after = self.state_dict()
        # updated_layers = [k for k in matching_keys if not torch.equal(state_dict_before[k], state_dict_after[k])]
        # # # 打印被更新的层的名称
        # print("Updated layers:")
        # for layer in updated_layers:
        #     print(layer)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        # backbone提取特征
        self.B = input_dict.count
        n = input_dict["obs"]["IDs"].shape[1]
        x = input_dict["obs"]["obs"]
        # state_ = input_dict["obs"]["state_"]
        # state_ = state_ / self.state_scale.to(state_.device)
        agent_boundary_map = x[:,1] * self.std[1] + self.mean[1] # boundary map
        self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 125
        other_obs = input_dict["obs"]["others_obs"]
        ids = input_dict["obs"]["IDs"]
        x = self.patch_embed(x)
        x = torch.cat((x.unsqueeze(1), other_obs), dim = 1)
        x = rearrange(x, 'B n p l -> (B n) p l', B = self.B)
        action_token = self.action_token + self.pos_embed[:, :2, :]
        action_token = action_token.expand(x.shape[0], -1, -1)
        state_token = self.state_token + self.pos_embed[:, 2, :]
        state_token = state_token.expand(x.shape[0], -1, -1)
        x = torch.cat((action_token, state_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, '(B n) p l -> B n p l', B = self.B)
        agent = x[:, 0]
        self.feature = agent # for value function
        others = self.others_norm(rearrange(x[:, 1:], 'B n p l -> B (n p) l'))
        for blk in self.ca_blocks:
            agent = blk(agent, others)
        x = agent
        x = self.norm(x) # x.shape = [B d l] = [32 1178 128]
        out_local_region = self.to_local_region(x[:,0])
        out_local_region = torch.where(self.IAM_mask, out_local_region, torch.tensor(-1e4).to(dtype=out_local_region.dtype, device=out_local_region.device))
        out_local_point = self.to_local_point(x[:,1])
        state_ = self.to_local_state(x[:,2])
        x = torch.cat((out_local_region, out_local_point), dim =1)
        return x, state 

    @override(TorchModelV2)
    def value_function(self) -> TensorType: 
        assert self.feature is not None, "must call forward() first"
        x = self.feature
        for blk in self.critic_blocks:
            x = blk(x)
        x = self.critic_norm(x)
        x = self.to_value(x[:,0])
        return torch.reshape(x, [-1])

    def actor_parameters(self): # 将actor的参数转化为列表
        params = []
        for module in self.actors:
            if isinstance(module, nn.Module):
                params.extend(list(module.parameters()))
            elif isinstance(module, nn.Parameter):
                params.append(module)
        return params
    
    def critic_parameters(self): # 返回所有local critic的函数
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.critic))
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_encoder(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.action_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
