import numpy as np
from gym.spaces import Box
from functools import reduce
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from marllib.marl.models.zoo.mlp.base_mlp import BaseMLP
from marllib.marl.models.zoo.encoder.cc_encoder import CentralizedEncoder
from torch.optim import Adam
from model.actor_cc_v3 import actor_cc
from model.transformer import Transformer
torch, nn = try_import_torch()
from einops import rearrange
from marllib.marl.algos.utils.centralized_Q import get_dim

class CentralizedCritic(actor_cc):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):

        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, **kwargs)

        self.q_flag = False
        self.ccvf_agent_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256, bias = True),
            nn.GELU(),
            nn.LayerNorm(256 , eps=1e-5,elementwise_affine=True),
            nn.Linear(256, 1, bias = True))


    def central_value_function(self, state, opponent_actions=None) -> TensorType: 
        assert self.features is not None, "must call forward() first"
        B = state.shape[0]
        data = {}
        obs_dim_list = []
        for key in self.full_obs_space:
            obs_dim_list.append(get_dim(self.full_obs_space[key].shape))
        states = state.split(obs_dim_list, dim=1)
        for index, key in enumerate(self.full_obs_space):
            data[key] = states[index].reshape(B, *self.full_obs_space[key].shape)

        per_obs = data["obs"]
        state_ = data["state_"]
        state_ = state_ / self.state_scale.to(state_.device)
        other_obs = data["others_obs"]
        ids = data["IDs"]
        n = data["IDs"].shape[1]
        input = torch.cat((per_obs.unsqueeze(1), other_obs), dim=1).reshape(-1, *per_obs.shape[1:])
        self.features = self.encoder(input)
        actor = self.features.reshape((B,-1,*self.features.shape[-3:]))[:,0]
        others = self.features.reshape((B,-1,*self.features.shape[-3:]))[:,1:]
        actor = rearrange(actor, 'B c h w -> B (h w) c')
        others = rearrange(others, 'B n c h w -> B n (h w) c')
        # for grid map
        grid_map = data["grid_map"]
        indices = [2*i+j+k for i in range(0,n) for j in range(0,2*n+1,2*n) for k in range(2)] 
        grid_map = grid_map[:, indices, :, :].view(-1, n, 4, 8, 8)
        grid_map = rearrange(grid_map, 'b n c h w -> (b n) c h w', c = 4, h = self.grid_size, w = self.grid_size)
        grid_feature = self.transformer(rearrange(grid_map, 'b c h w -> b (h w) c'))
        grid_feature = rearrange(grid_feature, '(b n) h c -> b n h c', n = n)
        actor = torch.cat([actor, grid_feature[:,0]], dim = -1)
        others = torch.cat([others, grid_feature[:,1:]], dim = -1)
        # attention
        out = self.agent_encoder[0]([actor,others], ids, state_)
        out = self.ccvf_agent_encoder(out)
        if self.q_flag:
            return torch.reshape(out, [-1, self.num_outputs])
        else:
            return torch.reshape(out, [-1])

    @override(actor_cc)
    def critic_parameters(self):
        critics = [
            self.encoder,
            self.agent_encoder[0],
            self.ccvf_agent_encoder
        ]
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), critics))

    def link_other_agent_policy(self, agent_id, policy):
        if agent_id in self.other_policies:
            if self.other_policies[agent_id] != policy:
                raise ValueError('the policy is not same with the two time look up')
        else:
            self.other_policies[agent_id] = policy

    def set_train_batch(self, batch):
        self._train_batch_ = batch.copy()

        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                try:
                    self._train_batch_[key] = torch.Tensor(value)
                except TypeError as e:
                    # print(f'key: {key} cannot be convert to Tensor')
                    pass

    def get_train_batch(self):
        return self._train_batch_

    def get_actions(self):
        return self(self._train_batch_)

    def update_actor(self, loss, lr, grad_clip):
        CentralizedCritic.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    def update_critic(self, loss, lr, grad_clip):
        CentralizedCritic.update_use_torch_adam(
            loss=loss,
            optimizer=self.critic_optimizer,
            parameters=self.critic_parameters(),
            grad_clip=grad_clip
        )

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in parameters if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()

    def __update_adam(self, loss, parameters, adam_info, lr, grad_clip, step, maximum=False):

        for p in self.parameters():
            p.grad = None

        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        total_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradients]))
        max_norm = grad_clip
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for g in gradients:
                g.detach().mul_(clip_coef.to(g.device))

        after_total_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradients]))
        if total_norm != after_total_norm:
            print(f'before clip norm: {total_norm}')
            print(f'after clip norm: {after_total_norm}')
        if after_total_norm - grad_clip > 1:
            raise ValueError(f'grad clip error!, after clip norm: {after_total_norm}, clip norm threshold: {grad_clip}')

        beta1, beta2 = 0.9, 0.999
        eps = 1e-05

        real_gradients = []

        if maximum:
            # for i, param in enumerate(parameters):
            for grad in gradients:
                # gradients[i] = -gradients[i]  # get maximize
                grad = -1 * grad
                real_gradients.append(grad)

            gradients = real_gradients

        m_v = []
        v_v = []

        if len(adam_info['m']) == 0:
            adam_info['m'] = [0] * len(gradients)
            adam_info['v'] = [0] * len(gradients)

        for i, g in enumerate(gradients):
            mt = beta1 * adam_info['m'][i] + (1 - beta1) * g
            vt = beta2 * adam_info['v'][i] + (1 - beta2) * (g ** 2)

            m_t_bar = mt / (1 - beta1 ** step)
            v_t_bar = vt / (1 - beta2 ** step)

            vector_to_parameters(
                parameters_to_vector([parameters[i]]) - parameters_to_vector(
                    lr * m_t_bar / (torch.sqrt(v_t_bar) + eps)),
                [parameters[i]],
            )

            m_v.append(mt)
            v_v.append(vt)

        step += 1

        adam_info['m'] = m_v
        adam_info['v'] = v_v

        return step
