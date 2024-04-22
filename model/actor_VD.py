# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.utils.framework import try_import_torch
from marllib.marl.models.zoo.mlp.base_mlp import BaseMLP
from model.actor_IL import actor_il
from marllib.marl.algos.utils.centralized_Q import get_dim
from marllib.marl.models.zoo.mixer import  VDNMixer
from model.Qmixer import QMixer
from einops import rearrange
torch, nn = try_import_torch()


class actor_vd(actor_il):

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

        # mixer:
        state_dim = (128,self.custom_config["num_agents"])
        if self.custom_config["algo_args"]["mixer"] == "qmix":
            self.mixer = QMixer(self.custom_config, state_dim)
        elif self.custom_config["algo_args"]["mixer"] == "vdn":
            self.mixer = VDNMixer()
        else:
            raise ValueError("Unknown mixer type {}".format(self.custom_config["algo_args"]["mixer"]))
        self.fc_group  = nn.Sequential(
            nn.LayerNorm(64*48),
            nn.Linear(64*48, 1024),
            nn.GELU(),
            nn.Linear(1024, 128),
        )

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        state = state.reshape(-1,state.shape[-1])
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
        grid_map = data["grid_map"]
        n = data["IDs"].shape[1]
        features = self.encoder(per_obs)
        actor = rearrange(features, 'B c h w -> B (h w) c')
        # for grid map     
        indices = [2*i+j+k for i in range(0,n) for j in range(0,2*n+1,2*n) for k in range(2)] 
        grid_map = grid_map[:, indices, :, :].view(-1, n, 4, 8, 8)[:,0]
        grid_feature = self.transformer(rearrange(grid_map, 'b c h w -> b (h w) c'))
        actor = torch.cat([actor, grid_feature], dim = -1)
        actor = rearrange(actor, 'b h c -> b (h c)')
        actor = self.fc_group(actor)
        state = rearrange(actor, '(b n) c -> b n c',n = n)
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer(all_agents_vf, state)

        return v_tot.flatten(start_dim=0)
