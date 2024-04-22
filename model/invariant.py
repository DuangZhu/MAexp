import numpy as np
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vit import ViT, Attention, PreNorm, Transformer, CrossAttention, FeedForward
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import random

def get_position_embedding(pos, hidden_dim, device = torch.device("cpu")):
    scaled_time = 2 * torch.arange(hidden_dim / 2) / hidden_dim
    scaled_time = 10000 ** scaled_time.to(device=device)
    scaled_time = pos.unsqueeze(1) / scaled_time.unsqueeze(0)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).to(device)

def get_explicit_position_embedding(n_embed_input=10, n_embed_output=8 , device = torch.device("cpu")):

    return nn.Embedding(n_embed_input, n_embed_output).to(device)

class AlterEncoder(nn.Module):
    def __init__(self, num_grids, input_dim, use_explicit_id = False, use_id_embedding = True, use_pos_embedding = True, depth = 1, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, dropout = 0., _multi_layer_cross_attn = True, _norm_sum = False,  _take_self = False,\
        use_intra_attn = True, use_self_attn = True, use_flatten_attn = False, hama=False):
        super().__init__()
        self.num_grids = num_grids
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_explicit_id = use_explicit_id 
        self.use_id_embedding = use_id_embedding  
        self.use_pos_embedding = use_pos_embedding  
        self._multi_layer_cross_attn = _multi_layer_cross_attn
        self._norm_sum = _norm_sum
        self._take_self = _take_self
        self.use_intra_attn = use_intra_attn
        self.use_self_attn = use_self_attn
        self.use_flatten_attn = use_flatten_attn
        self.attn_heads = heads
        self.hama = hama
        if self.use_pos_embedding and self.use_id_embedding:
            if self.use_explicit_id:
                self.random_embed = get_explicit_position_embedding(10, 8)
                self.encode_actor_net = nn.Linear(input_dim + hidden_dim + 8, hidden_dim)
                self.encode_other_net = nn.Linear(input_dim + hidden_dim + 8, hidden_dim)
            else:
                self.encode_actor_net = nn.Linear(input_dim + hidden_dim * 2, hidden_dim)
                self.encode_other_net = nn.Linear(input_dim + hidden_dim * 2, hidden_dim)
        elif self.use_pos_embedding or self.use_id_embedding:
            if self.use_explicit_id:
                self.random_embed = get_explicit_position_embedding(10, 8)
                self.encode_actor_net = nn.Linear(input_dim + 8, hidden_dim)
                self.encode_other_net = nn.Linear(input_dim + 8, hidden_dim)
            else:
                self.encode_actor_net = nn.Linear(input_dim + hidden_dim, hidden_dim)
                self.encode_other_net = nn.Linear(input_dim + hidden_dim, hidden_dim)
        else:
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim, hidden_dim)
        self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        self.other_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        if self.hama:
            self.hama_network = nn.ModuleList([
                        nn.Linear(num_grids * hidden_dim, hidden_dim),
                        Transformer(hidden_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.),
                        nn.Linear(hidden_dim, num_grids * hidden_dim)])
        elif self.use_flatten_attn:
            self.transformer = Transformer(hidden_dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
        else:
            self.spatial_attn_layers = nn.ModuleList([])
            self.agent_attn_layers = nn.ModuleList([])
            for _ in range(depth):
                self.spatial_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
                self.agent_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
        if self._multi_layer_cross_attn:
            self.cross_attn_layers = nn.ModuleList([])
            for _ in range(depth):
                self.cross_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout)),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
            self.out_norm = nn.LayerNorm(hidden_dim)
        elif self._norm_sum:
            self.fc_sum = PreNorm(hidden_dim, nn.Linear(hidden_dim, hidden_dim))           
        else:
            self.last_cross_attn = nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
            ])

    def forward(self, data, dgn_analyze=False):
        x, others = data
        B = x.shape[0]
        # print("alter_attn", x.shape)
        x = self.encode_actor(x)
        all = [x,]
        for i, y in enumerate(others):
            y = self.encode_other(y, i+1)
            all.append(y)
        num_agents = len(all)
        out = torch.stack(all, dim = 1) # B x num_agents x 64 x D
        if self.hama:
            fc_in, trans, fc_out = self.hama_network
            out = rearrange(out, "b n g d -> b n (g d)", b = B, n = num_agents, g = self.num_grids)
            out = fc_out(trans(fc_in(out)))
            out = rearrange(out, "b n (g d) -> b n g d", b = B, n = num_agents, g = self.num_grids)
        elif self.use_flatten_attn:
            for attn, ff in self.transformer.layers:
                out = rearrange(out, "b n g d -> b (n g) d", b = B, n = num_agents, g = self.num_grids)
                out = attn(out) + out
                out = rearrange(out, "b (n g) d -> b n g d", b = B, n = num_agents, g = self.num_grids)

                out = rearrange(out, "b n g d -> b (n g) d", b = B, n = num_agents, g = self.num_grids)
                out = ff(out) + out
                out = rearrange(out, "b (n g) d -> b n g d", b = B, n = num_agents, g = self.num_grids)
        else:
            for i in range(self.depth):
                if self.use_self_attn:
                    out = rearrange(out, "b n g d -> (b n) g d", b = B, n = num_agents, g = self.num_grids)
                    attn, fc = self.spatial_attn_layers[i]
                    out = attn(out) + out
                    out = fc(out) + out
                    out = rearrange(out, "(b n) g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
                
                if self.use_intra_attn:
                    if not self.use_self_attn:
                        out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
                    attn, fc = self.agent_attn_layers[i]
                    out = attn(out) + out
                    out = fc(out) + out
                out = rearrange(out, "(b g) n d -> b n g d", b = B, n = num_agents, g = self.num_grids)
        out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
        if self._multi_layer_cross_attn:
            x = out[:, :1, :] # 64B x 1 x D
            if num_agents > 1:
                others = self.out_norm(out[:, 1:, :]) # 64B x (n-1) x D
            else:
                others = []

            for i in range(self.depth):
                self_attn, ff1, norm, to_kv, cross_attn, ff2 = self.cross_attn_layers[i]
                x = rearrange(x, "(b g) n d -> (b n) g d", b = B, n = 1, g = self.num_grids) # B x 64 x D
                x = self_attn(x) + x
                x = ff1(x) + x
                x = rearrange(x, "(b n) g d -> (b g) n d", b = B, n = 1, g = self.num_grids) # 64B x 1 x D

                x = norm(x)
                if num_agents > 1:
                    k, v = to_kv(others).chunk(2, dim=-1)
                    if dgn_analyze:
                        out, attn_weight = cross_attn(x, k, v, ret=['attn'])
                        x = out + x
                    else:
                        x = cross_attn(x, k, v) + x
                x = ff2(x) + x
            out = x
        elif self._norm_sum:
            out = torch.sum(out, dim=1, keepdim=True)
            out = self.fc_sum(out)
        elif self._take_self:
            out = out[:, :1, :]
        else:
            norm, to_kv, cross_attn, ff= self.last_cross_attn
            out = norm(out)
            x = out[:, :1, :] # 64B x 1 x D
            others = out[:, 1:, :] # 64B x (n-1) x D
            if num_agents > 1:
                k, v = to_kv(others).chunk(2, dim=-1)
                if dgn_analyze:
                    out, attn_weight = cross_attn(x, k, v, ret=['attn'])
                    out = out + x
                else:
                    out = cross_attn(x, k, v) + x # # 64B x 1 x D
            else:
                out = x
            out = ff(out) + out
        out = rearrange(out, " (b g) n d -> n b g d", b = B, n = 1, g = self.num_grids)[0]
        if dgn_analyze:
            return out, rearrange(attn_weight, "(b g) h o n -> o b h n g", b = B, h = self.attn_heads, o = 1, n = num_agents - 1, g = self.num_grids)[0]
        return out
    
    def encode_actor(self, x):
        B = x.shape[0]
        if self.use_id_embedding:
            if self.use_explicit_id:
                self.agent_id = torch.ones((1,1), dtype=torch.long) * random.randint(0, 9)
                id_emb = self.random_embed(self.agent_id.to(x.device)).view(-1)
                id_emb = repeat(id_emb, "d -> b g d", b = B, g = self.num_grids)
            else:
                id_emb = get_position_embedding(0, self.hidden_dim, device = x.device)
                id_emb = repeat(id_emb, "d -> b g d", b = B, g = self.num_grids)
            x = torch.cat([x, id_emb], dim = -1)
        if self.use_pos_embedding:
            pos_emb = repeat(self.actor_pos_embedding, "() g d -> b g d", b = B)
            x = torch.cat([x, pos_emb], dim = -1)
        x = self.encode_actor_net(x)
        return x

    def encode_other(self, y, id):
        B = y.shape[0]
        if self.use_id_embedding:
            if self.use_explicit_id:
                self.agent_id = torch.ones((1,1), dtype=torch.long) * ((self.agent_id + 1) % 9)
                id_emb = self.random_embed(self.agent_id.to(y.device)).view(-1)
                id_emb = repeat(id_emb, "d -> b g d", b = B, g = self.num_grids)
            else:
                id_emb = get_position_embedding(id, self.hidden_dim, device = y.device)
                id_emb = repeat(id_emb, "d -> b g d", b = B, g = self.num_grids)
            y = torch.cat([y, id_emb], dim = -1)
        if self.use_pos_embedding:
            pos_emb = repeat(self.other_pos_embedding, "() g d -> b g d", b = B)
            y = torch.cat([y, pos_emb], dim = -1)
        y = self.encode_other_net(y)
        return y


class AgentEncoder(nn.Module):
    def __init__(self, num_grids, input_dim, depth = 1, hidden_dim=128, heads=4, dim_head=32, mlp_dim=128, dropout=0., pool = "mean"):
        super().__init__()
        self.num_grids = num_grids
        self.hidden_dim = hidden_dim
        self._pool = pool
        self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
        self.encode_other_net = nn.Linear(input_dim, hidden_dim)
        self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        self.other_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        self.attn_net = Transformer(hidden_dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
    
    def forward(self, data):
        x, others = data
        B = x.shape[0]
        x = self.encode_actor(x) # B x 64 x D
        for y in others:
            y = self.encode_other(y) # B x 64 x D
            x = torch.cat([x, y], dim = 1)
        z = self.attn(x)
        z = rearrange(z, "b (n g) d -> b n g d", g = self.num_grids)
        if self._pool == "mean":
            z = z.mean(dim=1)
        elif self._pool == "one":
            z = z[:, 0, :, :]
        else:
            raise NotImplementedError
        return z
    
    def encode_actor(self, x):
        fc = self.encode_actor_net
        x = fc(x) + self.actor_pos_embedding
        return x
    
    def encode_other(self, y):
        fc = self.encode_other_net
        y = fc(y) + self.other_pos_embedding
        return y
    
    def attn(self, x):
        x = self.attn_net(x)
        return x

class Invariant(nn.Module):
    def __init__(self, num_grids, input_dim, invariant_type = "attn_sum", hidden_dim=128, heads=4, dim_head=32, mlp_dim=128, dropout=0.):
        super().__init__()
        self.num_grids = num_grids
        self.hidden_dim = hidden_dim
        self.invariant_type = invariant_type
        if invariant_type == "attn_sum":
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim, hidden_dim)
            self.attn_net = nn.ModuleList([
                PreNorm(hidden_dim * 2, Attention(hidden_dim * 2, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(hidden_dim * 2, FeedForward(hidden_dim * 2, mlp_dim, dropout = dropout)),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ])
            # position embedding?
            self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
            self.other_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
            self.fc_max = nn.Linear(hidden_dim, hidden_dim)
            self.fc_sum = PreNorm(hidden_dim, nn.Linear(hidden_dim, hidden_dim))
        elif invariant_type == "attn_rnn":
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim, hidden_dim)
            self.attn_net = nn.ModuleList([
                PreNorm(hidden_dim * 2, Attention(hidden_dim * 2, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(hidden_dim * 2, FeedForward(hidden_dim * 2, mlp_dim, dropout = dropout)),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ])
            self.attn_rnn = nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
            ])
            # position embedding?
            self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
            self.other_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        elif invariant_type == 'attn_N':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim, hidden_dim)
            self.attn_net = Transformer(hidden_dim, depth = 1, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
            # position embedding?
            self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
            self.other_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        elif invariant_type == "mean":
            pass
        else:
            raise NotImplementedError
    
    def forward(self, x, others):
        B = x.shape[0]
        if self.invariant_type == "attn_sum":
            x = self.encode_actor(x) # B x 64 x D
            out_max = torch.zeros(B, self.num_grids, self.hidden_dim).to(x.device)
            out_sum = torch.zeros(B, self.num_grids, self.hidden_dim).to(x.device)
            for y in others:
                y = self.encode_other(y) # B x 64 x D
                z = torch.cat([x,y], dim=-1)
                z = self.attn(z)
                
                out_max = torch.max(out_max, z)
                out_sum = out_sum + z
            return 0.0 * self.fc_max(out_max) + 1.0 * self.fc_sum(out_sum)
        elif self.invariant_type == "attn_rnn":
            x = self.encode_actor(x) # B x 64 x D
            out = torch.zeros(B, self.num_grids, self.hidden_dim).to(x.device)
            for y in others:
                y = self.encode_other(y) # B x 64 x D
                z = torch.cat([x,y], dim=-1)
                z = self.attn(z)

                norm1, norm2, to_kv, cross_attn, ff = self.attn_rnn
                out = norm1(out)
                k, v = to_kv(norm2(z)).chunk(2, dim=-1)
                out = out + cross_attn(out, k, v)
                out = out + ff(out) 
            return out
        elif self.invariant_type == "attn_N":
            x = self.encode_actor(x) # B x 64 x D
            all = [x,]
            for y in others:
                y = self.encode_other(y)
                all.append(y)
            all = rearrange(torch.cat(all, dim = 1), "b (n g) d -> (b g) n d", g = self.num_grids)
            all = self.attn_net(all)
            all = all.mean(dim=1)
            all = rearrange(all, "(b g) d -> b g d", g = self.num_grids)
            return all
        elif self.invariant_type == "mean":
            for y in others:
                x = x+y
            return x / (len(others)+1)
        else:
            raise NotImplementedError
    
    def encode_actor(self, x):
        fc = self.encode_actor_net
        x = fc(x) + self.actor_pos_embedding
        return x
    
    def encode_other(self, y):
        fc = self.encode_other_net
        y = fc(y) + self.other_pos_embedding
        return y
    
    def attn(self, x):
        attn, ff, fc = self.attn_net
        x = attn(x) + x
        x = ff(x) + x
        x = fc(x)
        return x


class SingleAgentEncoder(nn.Module):
    def __init__(self, num_grids, input_dim, only_pos_and_goal = True,use_explicit_id = False, use_id_embedding = True, use_pos_embedding = True,use_add_id_embedding = False, depth = 2, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, dropout = 0., _multi_layer_cross_attn = True, _norm_sum = False,  _take_self = False,\
        use_intra_attn = True, use_self_attn = True, use_flatten_attn = False, hama=False, ):
        super().__init__()
        self.num_grids = num_grids
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_explicit_id = use_explicit_id 
        self.use_id_embedding = use_id_embedding
        self.use_pos_embedding = use_pos_embedding
        self.use_add_id_embedding = use_add_id_embedding
        self._multi_layer_cross_attn = _multi_layer_cross_attn
        self._norm_sum = _norm_sum
        self._take_self = _take_self
        self.use_intra_attn = use_intra_attn
        self.use_self_attn = use_self_attn
        self.use_flatten_attn = use_flatten_attn
        self.attn_heads = heads
        self.hama = hama
        self.feature_len = 36
        if self.use_add_id_embedding:
            self.pos_embedding = nn.Embedding(100, self.feature_len)
        # if only_pos_and_goal:
        #     self.share_dim = 16 # 跟邻居通讯的通道数,如果是完全ippo就是48，通讯pos则是16
        # else:
        # self.share_dim = 48
        if self.use_pos_embedding and self.use_id_embedding:
            self.encode_actor_net = nn.Linear(input_dim + hidden_dim * 2 + 4, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim + hidden_dim * 2 + 4, hidden_dim)
        elif self.use_add_id_embedding:
            self.encode_actor_net = nn.Linear(input_dim + 4, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim + 4, hidden_dim)
        elif self.use_pos_embedding or self.use_id_embedding:
            if self.use_explicit_id:
                self.random_embed = get_explicit_position_embedding(10, 8)
                self.encode_actor_net = nn.Linear(input_dim + 8, hidden_dim)
                self.encode_other_net = nn.Linear(self.share_dim + 8, hidden_dim)
            else:
                self.encode_actor_net = nn.Linear(input_dim + hidden_dim, hidden_dim)
                self.encode_other_net = nn.Linear(self.share_dim + hidden_dim, hidden_dim)
        else:
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(input_dim, hidden_dim)
        self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        self.other_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        if self.hama:
            self.hama_network = nn.ModuleList([
                        nn.Linear(num_grids * hidden_dim, hidden_dim),
                        Transformer(hidden_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.),
                        nn.Linear(hidden_dim, num_grids * hidden_dim)])
        elif self.use_flatten_attn:
            self.transformer = Transformer(hidden_dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
        else:
            self.spatial_attn_layers = nn.ModuleList([])
            self.agent_attn_layers = nn.ModuleList([])
            for _ in range(depth):
                self.spatial_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
                self.agent_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
        if self._multi_layer_cross_attn:
            self.cross_attn_layers = nn.ModuleList([])
            for _ in range(depth):
                self.cross_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout)),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
            self.out_norm = nn.LayerNorm(hidden_dim)
        elif self._norm_sum:
            self.fc_sum = PreNorm(hidden_dim, nn.Linear(hidden_dim, hidden_dim))           
        else:
            self.last_cross_attn = nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
            ])

    def forward(self, data, ids, state, locs=0, dgn_analyze=False ):
        B = data[0].shape[0]
        x, others = data
        # print("alter_attn", x.shape)
        x = self.encode_actor(x, ids, state).unsqueeze(1)
        y = self.encode_other(others, ids, state).reshape(B,-1,*x.shape[-2:])
        out = torch.cat((x,y), dim = 1)
        num_agents = out.shape[1]

        for i in range(self.depth):
            if self.use_self_attn:
                out = rearrange(out, "b n g d -> (b n) g d", b = B, n = num_agents, g = self.num_grids)
                attn, fc = self.spatial_attn_layers[i]
                out = attn(out) + out
                out = fc(out) + out
                out = rearrange(out, "(b n) g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
            
            if self.use_intra_attn:
                if not self.use_self_attn:
                    out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
                attn, fc = self.agent_attn_layers[i]
                out = attn(out) + out
                out = fc(out) + out
            out = rearrange(out, "(b g) n d -> b n g d", b = B, n = num_agents, g = self.num_grids)
        out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
        if self._multi_layer_cross_attn:
            x = out[:, :1, :] # 64B x 1 x D
            if num_agents > 1:
                others = self.out_norm(out[:, 1:, :]) # 64B x (n-1) x D
            else:
                others = []

            for i in range(self.depth):
                self_attn, ff1, norm, to_kv, cross_attn, ff2 = self.cross_attn_layers[i]
                x = rearrange(x, "(b g) n d -> (b n) g d", b = B, n = 1, g = self.num_grids) # B x 64 x D
                x = self_attn(x) + x
                x = ff1(x) + x
                x = rearrange(x, "(b n) g d -> (b g) n d", b = B, n = 1, g = self.num_grids) # 64B x 1 x D

                x = norm(x)
                if num_agents > 1:
                    k, v = to_kv(others).chunk(2, dim=-1)
                    if dgn_analyze:
                        out, attn_weight = cross_attn(x, k, v, ret=['attn'])
                        x = out + x
                    else:
                        x = cross_attn(x, k, v) + x
                x = ff2(x) + x
            out = x
        elif self._norm_sum:
            out = torch.sum(out, dim=1, keepdim=True)
            out = self.fc_sum(out)
        elif self._take_self:
            out = out[:, :1, :]
        else:
            norm, to_kv, cross_attn, ff= self.last_cross_attn
            out = norm(out)
            x = out[:, :1, :] # 64B x 1 x D
            others = out[:, 1:, :] # 64B x (n-1) x D
            if num_agents > 1:
                k, v = to_kv(others).chunk(2, dim=-1)
                if dgn_analyze:
                    out, attn_weight = cross_attn(x, k, v, ret=['attn'])
                    out = out + x
                else:
                    out = cross_attn(x, k, v) + x # # 64B x 1 x D
            else:
                out = x
            out = ff(out) + out
        out = rearrange(out, " (b g) n d -> n b g d", b = B, n = 1, g = self.num_grids)[0]
        if dgn_analyze:
            return out, rearrange(attn_weight, "(b g) h o n -> o b h n g", b = B, h = self.attn_heads, o = 1, n = num_agents - 1, g = self.num_grids)[0]
        
        return out
    
    def encode_actor(self, x, ids, state):
        B = x.shape[0]
        if self.use_id_embedding:
            id_emb = get_position_embedding(torch.zeros_like(ids[:,0]), self.hidden_dim, device = x.device) # 如果需要身份不一致去掉torch.zeros_like
            id_emb = repeat(id_emb, "b d -> b g d", b = B, g = self.num_grids)
            x = torch.cat([x, id_emb], dim = -1)
        if self.use_pos_embedding:
            pos_emb = repeat(self.actor_pos_embedding, "() g d -> b g d", b = B)
            x = torch.cat([x, pos_emb], dim = -1)
        states = repeat(state[:,0].unsqueeze(1), "b () d -> b g d", g = self.num_grids)
        x = torch.cat([x, states], dim = -1)
        if self.use_add_id_embedding:
            id_emb = self.pos_embedding(torch.zeros_like(ids[:,0]).int()).unsqueeze(1).expand(-1, 64, -1) # 如果需要身份不一致去掉torch.zeros_like
            x = self.encode_actor_net(x + id_emb)
        else:
            x = self.encode_actor_net(x)
        return x

    def encode_other(self, y, ids, state):
        y = rearrange(y, 'b n g c -> (b n) g c')
        ids = torch.ones_like(rearrange(ids[:,1:], 'b n -> (b n)')) # 如果需要身份不一致去掉torch.ones_like
        B = y.shape[0]
        if self.use_id_embedding:
            id_emb = get_position_embedding(ids, self.hidden_dim, device = y.device)
            id_emb = repeat(id_emb, "b d -> b g d", b = B, g = self.num_grids)
            y = torch.cat([y, id_emb], dim = -1)
        if self.use_pos_embedding:
            pos_emb = repeat(self.other_pos_embedding, "() g d -> b g d", b = B)
            y = torch.cat([y, pos_emb], dim = -1)
        states = repeat(state[:,1:].unsqueeze(2), "b n () d -> (b n) g d", g = self.num_grids)
        y = torch.cat([y, states], dim = -1)
        if self.use_add_id_embedding:
            id_emb = self.pos_embedding(ids.int()).unsqueeze(1).expand(-1, 64, -1)
            y = self.encode_actor_net(y + id_emb)
        else:
            y = self.encode_other_net(y)
        
        return y
    
    
class SingleAgent_CriticEncoder(nn.Module):
    def __init__(self, num_grids, input_dim, only_pos_and_goal = True,use_explicit_id = False, use_id_embedding = True, use_pos_embedding = True,use_add_id_embedding = False, depth = 2, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, dropout = 0., _multi_layer_cross_attn = True, _norm_sum = False,  _take_self = False,\
        use_intra_attn = True, use_self_attn = True, use_flatten_attn = False, hama=False, ):
        super().__init__()
        self.num_grids = num_grids
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_explicit_id = use_explicit_id 
        self.use_id_embedding = use_id_embedding
        self.use_pos_embedding = use_pos_embedding
        self.use_add_id_embedding = use_add_id_embedding
        self._multi_layer_cross_attn = _multi_layer_cross_attn
        self._norm_sum = _norm_sum
        self._take_self = _take_self
        self.use_intra_attn = use_intra_attn
        self.use_self_attn = use_self_attn
        self.use_flatten_attn = use_flatten_attn
        self.attn_heads = heads
        self.hama = hama
        self.feature_len = 36
        if self.use_add_id_embedding:
            self.pos_embedding = nn.Embedding(100, self.feature_len)
        if self.use_pos_embedding and self.use_id_embedding:
            self.encode_actor_net = nn.Linear(input_dim + hidden_dim * 2 + 4, hidden_dim)
        elif self.use_add_id_embedding:
            self.encode_actor_net = nn.Linear(input_dim + 4, hidden_dim)
        elif self.use_pos_embedding or self.use_id_embedding:
            if self.use_explicit_id:
                self.random_embed = get_explicit_position_embedding(10, 8)
                self.encode_actor_net = nn.Linear(input_dim + 8, hidden_dim)
            else:
                self.encode_actor_net = nn.Linear(input_dim + hidden_dim, hidden_dim)
        else:
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
        self.actor_pos_embedding = nn.Parameter(torch.randn(1, num_grids, hidden_dim))
        if self.hama:
            self.hama_network = nn.ModuleList([
                        nn.Linear(num_grids * hidden_dim, hidden_dim),
                        Transformer(hidden_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.),
                        nn.Linear(hidden_dim, num_grids * hidden_dim)])
        elif self.use_flatten_attn:
            self.transformer = Transformer(hidden_dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
        else:
            self.spatial_attn_layers = nn.ModuleList([])
            self.agent_attn_layers = nn.ModuleList([])
            for _ in range(depth):
                self.spatial_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
                self.agent_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
        if self._multi_layer_cross_attn:
            self.cross_attn_layers = nn.ModuleList([])
            for _ in range(depth):
                self.cross_attn_layers.append(nn.ModuleList([
                    PreNorm(hidden_dim, Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout)),
                    nn.LayerNorm(hidden_dim),
                    # nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                    # CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
                ]))
            # self.out_norm = nn.LayerNorm(hidden_dim)
        elif self._norm_sum:
            self.fc_sum = PreNorm(hidden_dim, nn.Linear(hidden_dim, hidden_dim))           
        else:
            self.last_cross_attn = nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
            ])


    def forward(self, data, state, locs=0, dgn_analyze=False ):
        B = data.shape[0]
        x = data
        num_agents = 1
        # print("alter_attn", x.shape)
        x = self.encode_actor(x, state).unsqueeze(1)
        for i in range(self.depth):
            if self.use_self_attn:
                out = rearrange(x, "b n g d -> (b n) g d", b = B, n = num_agents, g = self.num_grids)
                attn, fc = self.spatial_attn_layers[i]
                out = attn(out) + out
                out = fc(out) + out
                out = rearrange(out, "(b n) g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
            
            if self.use_intra_attn:
                if not self.use_self_attn:
                    out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
                attn, fc = self.agent_attn_layers[i]
                out = attn(out) + out
                out = fc(out) + out
            out = rearrange(out, "(b g) n d -> b n g d", b = B, n = num_agents, g = self.num_grids)
        out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
        if self._multi_layer_cross_attn:
            x = out[:, :1, :] # 64B x 1 x D
            if num_agents > 1:
                others = self.out_norm(out[:, 1:, :]) # 64B x (n-1) x D
            else:
                others = []

            for i in range(self.depth):
                self_attn, ff1, norm, ff2 = self.cross_attn_layers[i]
                x = rearrange(x, "(b g) n d -> (b n) g d", b = B, n = 1, g = self.num_grids) # B x 64 x D
                x = self_attn(x) + x
                x = ff1(x) + x
                x = rearrange(x, "(b n) g d -> (b g) n d", b = B, n = 1, g = self.num_grids) # 64B x 1 x D
                x = norm(x)
                x = ff2(x) + x
            out = x
        out = rearrange(out, " (b g) n d -> n b g d", b = B, n = 1, g = self.num_grids)[0]
        return out
    
    def encode_actor(self, x, state):
        B = x.shape[0]
        if self.use_id_embedding:
            id_emb = get_position_embedding(torch.zeros(B,device = x.device), self.hidden_dim, device = x.device) # 如果需要身份不一致去掉torch.zeros_like
            id_emb = repeat(id_emb, "b d -> b g d", b = B, g = self.num_grids)
            x = torch.cat([x, id_emb], dim = -1)
        if self.use_pos_embedding:
            pos_emb = repeat(self.actor_pos_embedding, "() g d -> b g d", b = B)
            x = torch.cat([x, pos_emb], dim = -1)
        states = repeat(state.unsqueeze(1), "b () d -> b g d", g = self.num_grids)
        x = torch.cat([x, states], dim = -1)
        if self.use_add_id_embedding:
            id_emb = self.pos_embedding(torch.zeros(B,device = x.device).int()).unsqueeze(1).expand(-1, 64, -1) # 如果需要身份不一致去掉torch.zeros_like
            x = self.encode_actor_net(x + id_emb)
        else:
            x = self.encode_actor_net(x)
        return x
