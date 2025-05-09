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
from model_vit.actor_mae_crossatt_IL import Crossat_actor_il
from marllib.marl.algos.utils.centralized_Q import get_dim
from marllib.marl.models.zoo.mixer import  VDNMixer
from model.Qmixer import QMixer
from einops import rearrange
torch, nn = try_import_torch()


class Crossat_vd(Crossat_actor_il):

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
        state_dim = (128, self.custom_config["num_agents"])
        if self.custom_config["algo_args"]["mixer"] == "qmix":
            self.mixer = QMixer(self.custom_config, state_dim)
        elif self.custom_config["algo_args"]["mixer"] == "vdn":
            self.mixer = VDNMixer()
        else:
            raise ValueError("Unknown mixer type {}".format(self.custom_config["algo_args"]["mixer"]))
        self.fc_group  = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        n = all_agents_vf.shape[-1]
        state = rearrange(state, 'b n c -> (b n) c',n = n)
        state = state.reshape(-1,state.shape[-1])
        B = state.shape[0]
        data = {}
        # 还原输入
        obs_dim_list = []
        for key in self.full_obs_space:
            obs_dim_list.append(get_dim(self.full_obs_space[key].shape))
        states = state.split(obs_dim_list, dim=1)
        for index, key in enumerate(self.full_obs_space):
            data[key] = states[index].reshape(B, *self.full_obs_space[key].shape)
        x = data["obs"]
        x = self.patch_embed(x)
        action_token = self.action_token + self.pos_embed[:, :2, :]
        action_token = action_token.expand(x.shape[0], -1, -1)
        state_token = self.state_token + self.pos_embed[:, 2, :]
        state_token = state_token.expand(x.shape[0], -1, -1)
        x = torch.cat((action_token, state_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        for blk in self.critic_blocks:
            x = blk(x)
        x = self.critic_norm(x)
        x = self.fc_group(x[:,1])
        x = rearrange(x, '(b n) c -> b n c',n = n)
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer(all_agents_vf, x)
        return v_tot.flatten(start_dim=0)
