import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box, Discrete
from gym import spaces
import sys
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from ray.tune import register_env
from tabulate import tabulate
import time
import torch
import math
import open3d as o3d
import yaml
import cv2
import logging
import copy
import matplotlib.pyplot as plt
import random
import env_utils.maze as maze
from env_utils.agent_ft_v2 import Agent_explorer
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from bulid_my_model_v2 import build_model
from einops import rearrange
import os
from PIL import Image
import argparse
from planning_method.ft_global_goal import ft_get_goal
import glob
import json

policy_mapping_dict = {
    "MAexp": { # scenario name
        "description": "explore in maze",
        "team_prefix": ("agent_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    }
}

# must inherited from MultiAgentEnv class
class Multiagent_exploration(MultiAgentEnv):  

    def __init__(self, exp_config): 
        """
        Initialize the multi-agent exploration environment.

        Args:
            exp_config (dict): Configuration for the environment.
        """
        self.config = exp_config
        self.agents = [f"agent_{i}" for i in range(self.config['num_agent'])]
        self.infos = {
            'Number_agent': self.config['num_agent'],
            'Steps': 0,
            'Episode': 0,
            'is_destroy': np.zeros(self.config['num_agent'], dtype=bool), 
            'Explore_ratio': 0.,
            '85% Coverage': 0.,
            '95% Coverage': 0.,
            '85% Overlap': 0.,
            '95% Overlap': 0.,
            '85% pathlen': 0.,
            '95% pathlen': 0.,
            'Total Reward': 0.,
            'Agents Rewards': np.zeros(self.config['num_agent']),
            'Rewards Variance': 0.,
            'init_state': np.zeros((self.config['num_agent'], 4)),
        }
        
        # Agent Configuration
        self.action_space = spaces.Tuple((
            Discrete(self.config['region']**2), 
            Box(low=np.array([-1.0,-1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32),
            ))  
        self.observation_space = GymDict({
            'obs': Box(low=-100., high=100., shape=(3, 128, 128), dtype=np.float32),
            'state_': Box(low=-100., high=1000., shape=(self.config['num_agent'],4), dtype=np.float32),
            'others_obs': Box(low=-100., high=100., shape=(self.config['num_agent']-1, 64, 128), dtype=np.float32),
            'commu': Box(low=-1., high=1., shape=(self.config['num_agent']-1, ), dtype=np.dtype('bool')),
            'IDs': Box(low=0, high=self.config['num_agent'], shape=(self.config['num_agent'],), dtype=np.dtype('int32')),
            })
 
        # Planning Config
        self.vel_range = torch.linspace(-self.config['max_a'], self.config['max_a'], self.config['vel_resolution'], device = self.config['device'])
        self.delta_range = torch.linspace(-self.config['max_sa'], self.config['max_sa'], self.config['delta_resolution'], device = self.config['device'])
        grid1, grid2 = torch.meshgrid(self.vel_range, self.delta_range)
        self.DWA_choice = torch.stack((grid1.reshape(-1), grid2.reshape(-1)), dim=1)
        self.DWA_choice = self.DWA_choice.unsqueeze(0).repeat(self.config['num_agent'],1,1)
        self.save_process = False
        if not self.config['is_train']:
            try:  
                with open(self.config['result_file'], 'r') as f: 
                    pass 
            except FileNotFoundError: 
                data = {'Episode':[], 'Steps':[], 'Explore_ratio':[], '85% Coverage':[],
                         '95% Coverage':[], '85% Overlap': [], '95% Overlap':[],
                           'Rewards Variance':[], '85% pathlen':[], '95% pathlen':[],
                           'Total Reward': [], 'init_state': [], 'Agents Rewards': []}
                with open(self.config['result_file'], 'w') as f:
                    json.dump(data, f, indent=4)

            self.capture = False
            self.textset = torch.load('./MAexp/testset/Final_testdata_randoms_test_10agent.pt')

    
    def reset(self):
        print(self.infos)
        if self.save_process:
            self.Process_data = {'Trajectory':[], 'explored_area':[]}
        if not self.config['is_train']:
            try: 
                with open(self.config['result_file'], 'r') as f: 
                    data = json.load(f) 
            except FileNotFoundError: 
                data = {} 
            if not self.infos['Episode'] == 0:
                for key, value in self.infos.items(): 
                    if key in data: 
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        elif torch.is_tensor(value):
                            value = value.cpu().tolist()
                        
                        data[key].append(value)  
                with open(self.config['result_file'], 'w') as f:
                    json.dump(data, f, indent=4)
                 
        if not self.config['is_train']:
            map_num = self.textset['test_id'][self.infos['Episode']].int()
        else:
            map_num = np.random.randint(0, self.config['training_map_num'])
        self.load_map(map_count=map_num) 
        print("map_ID:", self.config['scene'], self.config['map_list'][map_num])
        self.dw = self.map_w.cpu()/self.config['region']/2 # Discretize the width of half of the patch
        self.dh = self.map_h.cpu()/self.config['region']/2 # Discretize the height of half of the patch
        self.ft_pre_goals = np.zeros((self.config['num_agent'], 2), dtype = np.int32)
        self.explored_space = torch.empty((0, 3), device=self.config['device']) 
        self.scene_points = {
            'map_obstacles': self.map_obstacles,
            'map_freespace': self.map_freespace,
            'map_boundary': self.map_boundary,
            'map_real_w': self.map_w, 
            'map_real_h': self.map_h
        }
        self.all_area = len(self.map_freespace)
        self.pre_goal = None
        for key, value in self.scene_points.items():
            self.config[key] = value
        # Reward normalization based on the algorithm used
        self.reward_norm = 6000 / self.all_area * 0.01
        obs = {}
        self.agents_m = {}
        self.reward_list = {}
        self.path_len = torch.zeros(self.config['num_agent'], device = self.config['device'])
        self.explore_merged_map = torch.empty((0, 3), device=self.config['device'])
        '''
        init agent randomly
        '''
        self.env_vision = None
        if self.config['is_train']:
            ready = False
            while ready == False:
                self.agent_state = torch.zeros([self.config['num_agent'], 4], device=self.config['device'])
                self.agent_state[:,:2] = self.map_freespace[torch.randperm(len(self.map_freespace))[:self.config['num_agent']]][:, :2]
                self.agent_state[:,-1] = torch.rand(self.config['num_agent'], device = self.config['device']) * 2 * math.pi - math.pi
                self.is_collision = torch.zeros(self.config['num_agent'], device = self.config['device'])
                for i in range(self.config['num_agent']):
                    agent_ = Agent_explorer(i, self.agent_state[i], self.config) # type: ignore
                    self.agents_m[self.agents[i]] = agent_
                    self.reward_list[self.agents[i]] = {'explore_reward':[], 'overlap_reward':[], 'total_reward':[], 'time_reward':[]}
                self.env_vision = self.collect_agent_state()
                self.detect_collision(init = True)
                if self.env_vision['all_agent_collision'].any() == False:
                    ready = True
        else:
            self.agent_state = self.textset['text_set'][self.infos['Episode']].to(self.config['device'])
            self.is_collision = torch.zeros(self.config['num_agent'], device = self.config['device'])
            for i in range(self.config['num_agent']):
                agent_ = Agent_explorer(i, self.agent_state[i], self.config) # type: ignore
                self.agents_m[self.agents[i]] = agent_
                self.reward_list[self.agents[i]] = {'explore_reward':[], 'overlap_reward':[], 'total_reward':[], 'time_reward':[]}
            self.env_vision = self.collect_agent_state()
            self.detect_collision()
        print(self.agent_state)
        self.env_vision['all_agent_grid_map'] = self.calculate_grid_map()
        self.infos['Episode'] += 1
        self.infos['Steps'] = 0
        self.infos['local_Steps'] = 0
        self.infos['map_id'] = self.config['map_list'][map_num]
        diff = self.env_vision['all_agent_state'][:,:2].unsqueeze(1) - self.env_vision['all_agent_state'][:,:2].unsqueeze(0)
        self.dist = torch.sqrt(torch.sum(diff**2, dim=-1)) 
        self.dist[torch.eye(self.dist.size(0)).bool()] = float('nan')
        self.dist = self.dist[~torch.isnan(self.dist)].reshape(self.config['num_agent'], -1)  
        self.dist = self.dist < self.config['max_commu_dis']
        IDs = np.arange(0, self.config['num_agent'], dtype=np.int32)
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            agent.update_agent(self.env_vision) 
            ID_ = np.concatenate(([IDs[i]], IDs[:i], IDs[i+1:]))
            obs[self.agents[i]] = {"obs": agent.get_observation(self.dist[i]), 
                                   "state_": self.env_vision['all_agent_state'][ID_].cpu().numpy(),
                                    "commu": self.dist[i].cpu().numpy(),
                                    "IDs": ID_,
                                    "grid_map": self.env_vision['all_agent_grid_map'][i].cpu().numpy()
                                    }
        self.calculate_reward()
        self.threshold_reached_85 = False
        self.infos.update({
                'Explore_ratio': 0.,
                '85% Coverage': 0.,
                '95% Coverage': 0.,
                '85% Overlap': 0.,
                '95% Overlap': 0.,
                '85% pathlen': 0.,
                '95% pathlen': 0.,
                'Total Reward': 0.,
                'Agents Rewards': np.zeros(self.config['num_agent']),
                'Rewards Variance': 0.,
                'init_state': self.agent_state.cpu().numpy().tolist()
            })
        return obs
    
    def step(self): 
        self.infos['Steps'] += 1
        final_rewards = {}
        sum_rewards = np.zeros(self.config['num_agent'])
        obs = {}
        info = {}
        done = {'__all__': False}
        agents_bound_map = torch.zeros((self.config['num_agent'], 8000, 2), device=self.config['device'])
        for i, agent in enumerate(self.agents_m.values()):
            agents_bound_map[i,:len(agent.detected_bound_for_map)] = agent.detected_bound_for_map[:,:2]
            agent.local_step += 1
        # collect the image in this step before action    
        self.env_vision = env.collect_agent_state()
        if self.save_process:
                self.Process_data['Trajectory'].append(self.env_vision['all_agent_state'])
        if self.pre_goal is not None:
            region = self.get_region(self.pre_goal)
            self.env_vision['all_agent_grid_map'] = self.calculate_grid_map(region)
        else:
            self.env_vision['all_agent_grid_map'] = self.calculate_grid_map()
        IDs = np.arange(0, self.config['num_agent'], dtype=np.int32)
        for i, name in enumerate(self.agents_m):
            ID_ = np.concatenate(([IDs[i]], IDs[:i], IDs[i+1:]))
            obs[name] = {'obs': self.env_vision['all_agent_obs_image'][i]}
            obs[name]["state_"] = self.env_vision['all_agent_state'][ID_].cpu().numpy()
            obs[name]["commu"] = self.dist[i].cpu().numpy()
            obs[name]["IDs"] = ID_
            obs[name]["grid_map"] = self.env_vision['all_agent_grid_map'][i].cpu().numpy()
            obs[name]["all_agent_last_state"] = self.env_vision['all_agent_state'][i].cpu().numpy()
        # calculate global goal\
        input_ = env.merge_agent_map(self.env_vision)
        if env_config['algo'] == "ft_wma_rrt" or self.infos['local_Steps'] == 0:
            goal_mask = [False for agent_id in range(self.config['num_agent'])]
        else:
            goal_mask = [self.env_vision['all_agent_step'][agent_id] < self.config['local_step'] for agent_id in range(self.config['num_agent'])]
        goal = self.ft_compute_global_goal(input_, goal_mask, pre_goals = self.ft_pre_goals)
        self.pre_goal = copy.deepcopy(goal)
        # for step in range(self.config['local_step']):
        self.infos['local_Steps'] += 1
        _, action = self.DWA(goal, agents_bound_map)
        for i, agent in enumerate(self.agents_m.values()):
            agent.update_agent(self.env_vision)
            agent.step(action[i]) 
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            agent.get_observation(self.dist[i], goal)
            # agents_bound_map[i,:len(agent.detected_bound_for_map)] = agent.detected_bound_for_map[:,:2]
            
        rewards = self.calculate_reward()
        sum_rewards += rewards
        # if not self.config['is_train']:
        #     self.add_Goal(goal)
        #     self.render() 
        env_vision = self.collect_agent_state()
        self.detect_collision()
        del_l = env_vision['all_agent_state'][:,:2]-env_vision['all_agent_last_state'][:,:2]
        self.path_len += torch.sqrt(torch.sum(del_l**2, dim=1))
        if not self.threshold_reached_85 and self.infos['Explore_ratio'] > 0.85:
            print('reach_85%_coverage')
            self.threshold_reached_85 = True
            self.infos['85% Coverage'] = self.infos['local_Steps']
            maps = env_vision['all_agent_map'][:,0]
            merge_map = torch.sum(maps, dim = 0)
            area_a = torch.nonzero(merge_map).size(0)
            area_b = (merge_map > 1).sum().item()
            self.infos['85% Overlap'] = area_b/area_a
            self.infos['Agents Rewards'] += sum_rewards
            self.infos['85% pathlen'] = float(torch.sum(self.path_len).cpu())
        if self.infos['Explore_ratio'] > 0.95:
            done = {key: True for key in done}
            sum_rewards += np.ones(self.config['num_agent'])  * 100 *  self.reward_norm
            self.infos['Agents Rewards'] += sum_rewards
            self.infos['95% pathlen'] = float(torch.sum(self.path_len).cpu())
            self.calculate_final_metric(is_95 = True)
        elif all(env_vision['all_agent_collision']) or env.infos['local_Steps'] == env_config['local_step']*env_config['max_global_step']:
            if all(env_vision['all_agent_collision']):
                print('Destroy!!!')
                sum_rewards -= np.ones(self.config['num_agent']) * 200 * self.reward_norm
            done = {key: True for key in done}
            self.infos['Agents Rewards'] += sum_rewards
            self.calculate_final_metric(is_95 = False)
        else:
            self.infos['Agents Rewards'] += sum_rewards
        # todo 增加善后
        diff = env_vision['all_agent_state'][:,:2].unsqueeze(1) - env_vision['all_agent_state'][:,:2].unsqueeze(0)
        self.dist = torch.sqrt(torch.sum(diff**2, dim=-1)) 
        self.dist[torch.eye(self.dist.size(0)).bool()] = float('nan')
        self.dist = self.dist[~torch.isnan(self.dist)].reshape(self.config['num_agent'], -1)  
        self.dist = self.dist < self.config['max_commu_dis']
        
        # others_obs = self.env_vision['all_agent_map'].to(next(encoder.parameters()).device)
        # others_obs = encoder(others_obs).detach()
        for i, name in enumerate(self.agents_m):
            final_rewards[name] = sum_rewards[i]
            info[name] = {}
            done[name] = False
            
        return obs, final_rewards, done, goal
    
    def ft_compute_global_goal(self, inputs, goal_mask, pre_goals):
        goals = ft_get_goal(self, inputs, goal_mask, pre_goals = pre_goals)
        for i, name in enumerate(self.agents_m):
            if not goal_mask[i] or 'utility' in self.config['algo']:
                self.ft_pre_goals[i] = np.array(goals[i], dtype=np.int32)
                self.agents_m[name].local_step = 0
        goals = torch.from_numpy(goals).to(self.config['device'])
        return goals
            
    def calculate_grid_map(self, action = None):
        out_pos = torch.zeros([self.config['num_agent'], self.config['num_agent']*2, self.config['region'], self.config['region']], device = self.config['device'])
        out_goal = torch.zeros([self.config['num_agent'], self.config['num_agent']*2, self.config['region'], self.config['region']], device = self.config['device'])
        loc = self.env_vision['all_agent_state'][:,:2]
        loc_x = torch.div(loc[:, 0], self.dw * 2, rounding_mode='floor')
        loc_y = torch.div(loc[:, 1], self.dh * 2, rounding_mode='floor')
        loc = torch.stack((loc_x,loc_y), dim = 1).unsqueeze(-1).expand(-1,-1,self.config['region']) # self.num_aget , 2
        I = torch.arange(self.config['region'], device = self.config['device']).unsqueeze(0).unsqueeze(0).expand_as(loc)
        loc = I-loc
        xpos = loc[:,0,:].unsqueeze(0).expand(self.config['num_agent'], -1, -1) # self.num_aget, x, 8
        out_pos[:,0:2*self.config['num_agent']:2,:,:] = xpos.unsqueeze(-1).expand(*xpos.shape, self.config['region'])
        ypos = loc[:,1,:].unsqueeze(0).expand(self.config['num_agent'], -1, -1) # self.num_aget, y, 8
        # out_pos[:,1:2*self.num_agent:2,:,:] = ypos.unsqueeze(-1).expand(*ypos.shape, self.region)
        out_pos[:,1:2*self.config['num_agent']:2,:,:] = ypos.unsqueeze(-2).expand(*ypos.shape[:-1], self.config['region'], self.config['region'])
        out_pos += torch.ones_like(out_pos, device=self.config['device'])*(self.config['region'] - 1) #(0-14)
        if action is None:
            out_goal = out_pos
        else:
            x = action[:, 0]
            y = action[:, 1]
            region = torch.stack((x,y), dim = 1).unsqueeze(-1).expand(-1,-1,self.config['region'])
            region = I-region
            xpos = region[:,0,:].unsqueeze(0).expand(self.config['num_agent'], -1, -1) # self.num_aget, x, 8
            out_goal[:,0:2*self.config['num_agent']:2,:,:] = xpos.unsqueeze(-1).expand(*xpos.shape, self.config['region'])
            ypos = region[:,1,:].unsqueeze(0).expand(self.config['num_agent'], -1, -1) # self.num_aget, y, 8
            # out_goal[:,1:2*self.num_agent:2,:,:] = ypos.unsqueeze(-1).expand(*ypos.shape, self.region)
            out_goal[:,1:2*self.config['num_agent']:2,:,:] = ypos.unsqueeze(-2).expand(*ypos.shape[:-1], self.config['region'], self.config['region'])
            out_goal += torch.ones_like(out_pos, device=self.config['device'])*(self.config['region'] - 1) #(0-14)
        out_pos_ = copy.deepcopy(out_pos)
        out_goal_ = copy.deepcopy(out_goal)
        IDs = np.arange(0, self.config['num_agent']*2, dtype=np.int32)
        for a in range(1, self.config['num_agent']):
            ID_ = np.concatenate((IDs[2*a:2*(a+1)],IDs[:2*a], IDs[2*(a+1):]))
            out_pos_[a] = out_pos[a][ID_]
            out_goal_[a] = out_goal[a][ID_]
        return torch.cat((out_pos_, out_goal_), dim = 1) # 前半为位置，后半为全局坐标
    

    def get_region(self, actions):
        a = torch.tensor([self.dw*2, self.dh*2], device = self.config['device'])
        region = torch.floor(actions.to(torch.float32)/a).to(torch.int32)
        return region
    
    def local_to_global(self, local_point, car_pos):
        '''
        Calcluate the global position of the goal from local map
        '''
        center = torch.tensor([self.map_w / 2, self.map_h / 62.5])
        dx = local_point[:, 0] - center[0]
        dy = local_point[:, 1] - center[1]
        x = dx*torch.cos(-car_pos[:, -1]) + dy*torch.sin(-car_pos[:, -1])
        y = dy*torch.cos(-car_pos[:, -1]) - dx*torch.sin(-car_pos[:, -1])
        return torch.stack((car_pos[:, 0] + x, car_pos[:, 1] + y), dim=1).to(self.config['device'])

    def calculate_final_metric(self, is_95):
        if is_95:
            print('reach_95%_coverage')
            self.infos['95% Coverage'] = self.infos['local_Steps']
            maps = self.env_vision['all_agent_map'][:,0]
            merge_map = torch.sum(maps, dim = 0)
            area_a = torch.nonzero(merge_map).size(0)
            area_b = (merge_map > 1).sum().item()
            self.infos['95% Overlap'] = area_b/area_a
        self.infos['Total Reward']  = np.sum(self.infos['Agents Rewards'])
        self.infos['Rewards Variance'] = np.sqrt(np.var(self.infos['Agents Rewards']))
        if self.save_process:
            self.Process_data['explored_area'].append(self.explore_merged_map)
            np.save('/remote-home/ums_zhushaohao/new/2024/visual_data/rrt/random3/'+str(self.infos['Episode'])+'.npy', self.Process_data)
        
    
    def calculate_reward(self):
        rewards = np.zeros(self.config['num_agent'])
        """
        探索重叠面积惩罚
        """
        overlap = torch.zeros(self.config['num_agent'], self.config['num_agent'], device = self.config['device'], dtype=torch.int)
        for i in range(self.config['num_agent']):
            for j in range(i+1, self.config['num_agent']):
                cars_overlap = torch.cat((self.agents_m[self.agents[i]].freespace_selected_points_mat,
                                           self.agents_m[self.agents[j]].freespace_selected_points_mat), dim=0)
                overlap[i,j] = cars_overlap.size()[0] - torch.unique(cars_overlap, dim = 0, sorted = False).size()[0] 
        overlap_reward = (overlap.T + overlap).sum(0)

        """
        探索覆盖奖励，每一个智能体新探索区域的面积
        """
        for name in self.agents:
            map_overlap = torch.cat((self.agents_m[name].freespace_selected_points_mat,
                                           self.explore_merged_map), dim=0)
            self.reward_list[name]['explore_reward'].append(self.agents_m[name].freespace_selected_points_mat.size()[0] - map_overlap.size()[0]
                                                            + torch.unique(map_overlap, dim = 0, sorted = False).size()[0])
            self.reward_list[name]['overlap_reward'].append(overlap_reward[int(name.split('_')[-1])].cpu())
            self.reward_list[name]['time_reward'].append((-(self.all_area-self.explore_merged_map.shape[0])/self.all_area))
        # 更新 merge map
        for i, name in enumerate(self.agents):
            self.explore_merged_map = torch.cat((self.agents_m[name].freespace_selected_points_mat, self.explore_merged_map), dim = 0)
            # rewards[i] = float(self.reward_list[name]['time_reward'][-1] + self.reward_list[name]['explore_reward'][-1]) * self.reward_norm
            rewards[i] = float(self.reward_list[name]['time_reward'][-1] + self.reward_list[name]['explore_reward'][-1] * 0.3 - self.reward_list[name]['overlap_reward'][-1] * 0.01)  * self.reward_norm
        # self.explore_merged_map = torch.cat(self.explore_merged_map, dim=0)
        try:
            self.explore_merged_map = torch.unique(self.explore_merged_map, dim = 0, sorted = False)
        except RuntimeError as e:
            print(f"explore_merged_map shape: {self.explore_merged_map.shape}") 
            print(f"explore_merged_map device: {self.explore_merged_map.device}") 
            print(f"explore_merged_map dtype: {self.explore_merged_map.dtype}") 
            print(f"explore_merged_map has nan: {torch.any(torch.isnan(self.explore_merged_map))}") 
            raise e
        self.infos['Explore_ratio'] = self.explore_merged_map.shape[0]/self.all_area
        return rewards

    def close(self):
        pass

    def collect_agent_state(self):
        """
        收集智能体状态
        """
        all_agent_state = []
        all_agent_model_mats = []
        all_agent_map = []
        all_grid_map = []
        all_agent_local_step = []
        all_agent_obs_image = []
        if self.env_vision is not None:
            all_agent_last_state = self.env_vision['all_agent_state'].clone()
        else:
            all_agent_last_state = None
        for name in self.agents_m:
            agent = self.agents_m[name]
            all_agent_state.append(agent.agent_state)
            all_agent_model_mats.append(agent.car_model_mat)
            all_agent_map.append(agent.obs)
            all_agent_local_step.append(agent.local_step)
            all_agent_obs_image.append(agent.out)
            if agent.explored_space is not None:
                self.explored_space = torch.cat((self.explored_space, agent.explored_space), dim=0)
            self.explored_space = torch.unique(self.explored_space, dim = 0, sorted = False)
        all_agent_state = torch.cat(all_agent_state, dim=0).reshape(-1,4)
        all_agent_model_mats = torch.cat(all_agent_model_mats, dim=0)
        all_agent_map = torch.stack(all_agent_map)
        # if all_agent_obs_image[0] is not None:
        #     all_agent_obs_image = torch.stack(all_agent_obs_image)
        env_vision = {
            'all_agent_state': all_agent_state, 
            'all_agent_model_mats': all_agent_model_mats,
            'all_agent_collision': self.is_collision,
            'all_agent_map': all_agent_map,
            'all_agent_grid_map': all_grid_map,
            'all_agent_step': all_agent_local_step,
            'all_agent_obs_image': all_agent_obs_image,
            'all_agent_last_state': all_agent_last_state
        }
        return env_vision
    
    def merge_agent_map(self, agent_info):
        all_agent_map = agent_info['all_agent_map']
        merged_explore_map = torch.max(all_agent_map[:,0], dim=0).values
        merged_obstacle_map = torch.max(all_agent_map[:,1], dim=0).values
        locations = []
        for name in self.agents_m:
            agent = self.agents_m[name]
            # agent.obs = torch.stack([merged_explore_map, merged_obstacle_map]) # 可以等以后可视化的时候再使用
            locations.append((int(agent.agent_state[0]),int(agent.agent_state[1])))
        
        inputs = {
            'map_pred' : merged_obstacle_map.cpu().numpy(),
            'exp_pred' : merged_explore_map.cpu().numpy(),
            'locations' : locations
        }
        return inputs
    
    def detect_collision(self, init = False):
        """
        检测小车是否发生碰撞
        """
        if self.config['use_all_points_collision']:
            dis_env = torch.norm(self.env_vision['all_agent_model_mats'][:,:2].unsqueeze(1)-self.map_boundary[:,:2].unsqueeze(0), dim = 2)
            dis_car = torch.norm(self.env_vision['all_agent_model_mats'][:,:2].unsqueeze(1)-self.env_vision['all_agent_model_mats'][:,:2].unsqueeze(0), dim = 2)
            for i in range(0, dis_car.shape[0], self.config['agent_resolution']**2):
                dis_car[i:i+self.config['agent_resolution']**2, i:i+self.config['agent_resolution']**2] = float('inf')
            dis_env = dis_env.reshape(self.config['num_agent'], -1)
            dis_car = dis_car.reshape(self.config['num_agent'], -1)
            min_dis,_ = torch.min(torch.cat((dis_env,dis_car), dim = -1),dim = 1)
        else:
            dis_env = torch.norm(self.env_vision['all_agent_state'][:,:2].unsqueeze(1)-self.map_boundary[:,:2].unsqueeze(0), dim = 2)
            dis_car = torch.norm(self.env_vision['all_agent_state'][:,:2].unsqueeze(1)-self.env_vision['all_agent_state'][:,:2].unsqueeze(0), dim = 2)
            diag_indices = torch.arange(dis_car.shape[0])
            dis_car[diag_indices, diag_indices] = float('inf')
            min_dis,_ = torch.min(torch.cat((dis_env,dis_car), dim = 1),dim = 1)
        # print(min_dis)
        if init:
            self.env_vision['all_agent_collision'] = min_dis < self.config['collision_threshold'] * 3
        else:
            self.env_vision['all_agent_collision'] = min_dis < self.config['collision_threshold']
        self.is_collision = self.env_vision['all_agent_collision']   

    def get_agents_linear_matrix(self, states, actions):
        '''
        state:[(n d) 4]
        action: [(n,d) 2]
        '''
        A = torch.zeros((len(actions), 4, 4), device = self.config['device'])
        B = torch.zeros((len(actions), 4, 2), device = self.config['device'])
        C = torch.zeros((len(actions), 4), device=self.config['device'])
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 1.0
        A[:, 3, 3] = 1.0
        A[:, 0, 2] = self.config['DT'] * torch.cos(states[:,3])
        A[:, 0, 3] = - self.config['DT'] * states[:,2] * torch.sin(states[:,3])
        A[:, 1, 2] = self.config['DT'] * torch.sin(states[:,3])
        A[:, 1, 3] = self.config['DT'] * states[:,2] * torch.cos(states[:,3])
        A[:, 3, 2] = self.config['DT'] * torch.tan(actions[:,1]) / self.config['agent_wide']

        B[:, 2, 0] = self.config['DT']
        B[:, 3, 1] = self.config['DT'] * states[:,2] / (self.config['agent_wide'] * torch.cos(actions[:,1]) ** 2)

        C[:, 0] = self.config['DT'] * states[:,2] * torch.sin(states[:,3]) * states[:,3]
        C[:, 1] = - self.config['DT'] * states[:,2] * torch.cos(states[:,3]) * states[:,3]
        C[:, 3] = - self.config['DT'] * states[:,2] * actions[:,1] / (self.config['agent_wide'] * torch.cos(actions[:,1]) ** 2)

        return A, B, C

    
    def DWA(self, target_state, agents_bound_map, sim_step=3):
        """
        以下变量可设置为config变量
        """
        best_reward = torch.full((self.config['num_agent'],), float('-inf'), device = self.config['device'])
        best_action = torch.zeros((self.config['num_agent'], 2), device = self.config['device'] )
        action = rearrange(self.DWA_choice, 'n l d -> (n l) d')
        z = self.env_vision['all_agent_state'].unsqueeze(1).repeat(1,int(len(action)/self.config['num_agent']),1)
        z = rearrange(z, 'n l d -> (n l) d').unsqueeze(-1)
        for t in range(sim_step):
            A, B, C = self.get_agents_linear_matrix(z[:,:,0], action) # type: ignore
            z = torch.bmm(A, z) + torch.bmm(B, action.unsqueeze(-1)) + C.unsqueeze(-1)
        z = z.reshape(self.config['num_agent'], -1, 4)
        # To car center
        x = (z[:,:,0] + 0.5 * self.config['agent_length']*torch.cos(z[:,:,3])).reshape(self.config['num_agent'], -1)
        y = (z[:,:,1] + 0.5 * self.config['agent_length']*torch.sin(z[:,:,3])).reshape(self.config['num_agent'], -1)
        target_state = target_state.unsqueeze(1)
        target_reward = -torch.norm(torch.stack((x,y),dim=2) - target_state, dim = 2) # 最短距离 
        dis_obs = torch.norm(torch.stack((x,y),dim=2).unsqueeze(1) - agents_bound_map.unsqueeze(2), dim = -1)
        distance_to_obstacle = dis_obs.min(dim=1)[0]
        obstacle_penalty = torch.zeros_like(distance_to_obstacle, device=self.config['device'])
        mask = distance_to_obstacle < self.config['multi_para'] * self.config['collision_threshold']
        obstacle_penalty[mask] = torch.log(distance_to_obstacle[mask]/(self.config['multi_para'] * self.config['collision_threshold']))
        predict_reward = target_reward + 100 * obstacle_penalty 
        
        best_values, idxs = torch.max(predict_reward, dim=1)
        action = action.reshape(self.config['num_agent'], -1, 2)
        best_action = action[torch.arange(action.size(0)), idxs]
        return best_values, best_action         
    
    def add_Goal(self,goal):
        points = np.array(goal.cpu())
        # 创建一个空的列表来存储所有圆上的点
        circle_points = []
        # 循环遍历每个点
        for point in points:
        # 在XY平面上创建一个半径为1的圆
            theta1 = np.linspace(0, 2 * np.pi, 100)
            circle_x = point[0] + np.cos(theta1)*0.2*self.config['map_resolution']
            circle_y = point[1] + np.sin(theta1)*0.2*self.config['map_resolution']
            circle_z = np.zeros_like(theta1)

            # 将圆上的点添加到列表中
            circle_points.extend(np.column_stack((circle_x, circle_y, circle_z)))

        # 将列表转换为NumPy数组
        circle_points = np.array(circle_points)
        points_3d = circle_points
        
        self.goal_point_cloud.points = o3d.utility.Vector3dVector(points_3d + np.array([0, 0, 0.03]))
        colors = np.zeros((len(points_3d), 3))  # 创建一个与点数量相同的全零数组

        # 设置前100个点的颜色为[1, 0, 0]
        colors[:len(points_3d)//3] = [245/255, 108/255, 108/255]
        # 设置中间100个点的颜色为[0, 1, 0]
        colors[len(points_3d)//3:len(points_3d)//3*2] = [253/255, 210/255, 224/255]

        # 设置最后100个点的颜色为[0, 0, 1]
        colors[len(points_3d)//3*2:] = [245/255, 150/255, 125/255]
        self.goal_point_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.goal_point_cloud)     


    def render(self):
        # [57/255, 197/255, 187/255]
        self.explored_space_o3d.points = o3d.utility.Vector3dVector(self.explored_space.cpu()+ torch.tensor([0, 0, 0.01]))
        self.explored_space_o3d.paint_uniform_color([197/255, 237/255, 96/255])
        self.map_freespace_o3d.paint_uniform_color([0.96, 1, 0.71])
        side_space_o3d = o3d.geometry.PointCloud()
        side_space_o3d.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0],
                                                                     [0 , int(torch.max(self.map_obstacles[:,1])), 0],
                                                                     [int(torch.max(self.map_obstacles[:,0])), 0, 0],
                                                                     [int(torch.max(self.map_obstacles[:,0])), int(torch.max(self.map_obstacles[:,1])), 0]])) 
        side_space_o3d.paint_uniform_color([0.6, 0.6, 0.6])
        self.vis.add_geometry(side_space_o3d)
        self.vis.update_geometry(self.explored_space_o3d)
        # enter = self.env_vision['all_agent_state'].cpu().numpy()
        # self.add_Circle(enter)
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            self.car_model[i].points = o3d.utility.Vector3dVector(agent.car_model_mat.cpu() + torch.tensor([0, 0, 0.05]))
            if i == 0:
                self.car_model[i].paint_uniform_color([245/255, 108/255, 108/255])
            elif i == 1:
                self.car_model[i].paint_uniform_color([253/255, 199/255, 209/255])
            elif i == 2:
                self.car_model[i].paint_uniform_color([245/255, 150/255, 125/255])
            self.vis_bound[i].points = o3d.utility.Vector3dVector(agent.detected_bound.cpu() + torch.tensor([0, 0, 0.01]))
            self.vis_bound[i].paint_uniform_color([0.6, 0.6, 0.6]) # 已经探索障碍
            self.vis.update_geometry(self.car_model[i])
            self.vis.update_geometry(self.vis_bound[i])
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.capture:
            save_path = '/home/shaohao/Documents/MAexp/img'
            subfolders = ['all', 'agent_0', 'agent_1', 'agent_2']
            if not os.path.exists(save_path):
                # 如果文件夹不存在，则创建它
                os.makedirs(save_path)
                print(f"Folder '{save_path}' created!")
                for subfolder in subfolders:
                    os.makedirs(os.path.join(save_path, subfolder))
            else:
                print(f"Folder '{save_path}' already exists.")

            self.save_screenshot()
            for i in range(self.config['num_agent']):
                img = self.env_vision['all_agent_map'][i,0].cpu().numpy()
                img = np.where(img==1,255,0).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(f"/home/shaohao/Documents/MAexp/img/agent_{str(i)}/screenshot_{self.infos['local_Steps']}.png")
        return True
   

    def save_screenshot(self):
        filename = f"/home/shaohao/Documents/MAexp/img/all/screenshot_{self.infos['local_Steps']}.png"
        # 保存当前窗口为图像文件
        self.vis.capture_screen_image(filename)
        # 输出保存成功的消息
        print(f"Saved screenshot: {filename}")   

  
    def load_map(self, map_count = None, id = None):
        """
        加载点云格式的地图模型 
        """
        if map_count is not None:
            self.map_freespace = torch.from_numpy(np.load("./map/"+self.config['scene']+"/"+self.config['map_list'][map_count]+"_freespace.npy")).float().to(self.config['device'])
            self.map_obstacles = torch.from_numpy(np.load("./map/"+self.config['scene']+"/"+self.config['map_list'][map_count]+"_obstacles.npy")).float().to(self.config['device'])
            self.map_boundary = torch.from_numpy(np.load("./map/"+self.config['scene']+"/"+self.config['map_list'][map_count]+"_boundary.npy")).float().to(self.config['device'])
            if self.config['scene'] in ['maze', 'random', 'maze9', 'random3','maze_4_change']:
                self.map_w = torch.tensor(125).to(self.config['device'])
                self.map_h = torch.tensor(125).to(self.config['device'])
            elif self.config['scene'] == 'indoor':
                map = np.load("./map/indoor/"+self.config['map_list'][map_count]+"_map.npy")
                self.map_w = torch.tensor(map.shape[0]/2 * self.config['map_resolution'])
                self.map_h = torch.tensor(map.shape[1]/2 * self.config['map_resolution'])
            elif self.config['scene'] == 'outdoor':
                self.map_w = torch.max(self.map_Obstacles[map_count][:,0])
                self.map_h = torch.max(self.map_Obstacles[map_count][:,1])
        elif id is not None:
            if self.config['scene'] in ['maze', 'random', 'maze9']:
                self.map_w = torch.tensor(125).to(self.config['device'])
                self.map_h = torch.tensor(125).to(self.config['device'])
                self.map_freespace = torch.from_numpy(np.load("./map/"+self.config['scene']+"/map"+str(id)+"_freespace.npy")).float().to(self.config['device'])
                self.map_obstacles = torch.from_numpy(np.load(".map/"+self.config['scene']+"/map"+str(id)+"_obstacles.npy")).float().to(self.config['device'])
                self.map_boundary = torch.from_numpy(np.load("./map/"+self.config['scene']+"/map"+str(id)+"_boundary.npy")).float().to(self.config['device'])

    
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.config['num_agent'],
            "episode_limit": self.config['max_global_step'],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def set_gpu_device():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

def get_config():
    parser = argparse.ArgumentParser(
        description='Maexp', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str, default = "cuda", choices = ["cuda", "cpu"], help="will use GPU to train; or else will use CPU;")
    parser.add_argument('--num_agent', type=int, default=3, help="the number of agent in the swarm")
    parser.add_argument('--is_train', action='store_true', default=False, help="by default True, trian the policy, else test.")
    parser.add_argument("--algo", type=str, default = "mappo", choices = ["ippo", "itrpo", "mappo", "matrpo", "vdppo", "vda2c"], help="choice an marl algorithms to train the policy")
    parser.add_argument("--yaml_file", type=str, default = './yaml/random_ft.yaml', help="the yaml file for the experiment parameter")
    parser.add_argument('--make_data', action='store_true', default=True, help="whether make pretrain data") # check:whether make data
    parser.add_argument("--result_file", type=str, default = './paper3_result/a.json', help="the yaml file for the experiment parameter")
    
    """
    map config
    """
    parser.add_argument('--training_map_num', type=int, default=1, help="the number of different maps in training")
    parser.add_argument('--map_resolution', type=float, default=1.5, help="the resolution of the maps in training")
    parser.add_argument('--region', type=int, default=8, help="the number of disperse region will be divided of a map")
    parser.add_argument('--max_global_step', type=int, default=20, help="the number of global step in an exploration episode")
    parser.add_argument("--scene", type=str, default = "outdoor", choices = ["random", "maze", "indoor", "outdoor"], help="choice the scene to explore")
    parser.add_argument('--map_list', default = None, help="the maps used in experiment")
    """
    Agent Configuration
    """
    parser.add_argument('--DT', type=float, default=0.1, help="second,the time of one step")
    parser.add_argument("--max_commu_dis", type=float, default = 80.0, help="the max distence for communication between agents")
    parser.add_argument('--agent_length', type=float, default = 1.5, help="meter, the length of the agent")
    parser.add_argument('--agent_wide', type=float, default = 1.0, help="meter, the wide of the agent")
    parser.add_argument('--max_speed', type=float, default = 6.0, help="m/s, the max speed of the agent")
    parser.add_argument('--use_all_points_collision', default = False, help="by default False, only use the center of mass to detect collision, else the whole agent.")
    parser.add_argument('--agent_resolution', type=int, default=5, help="resolution of an agent point cloud")
    parser.add_argument('--is_lidar', default=True, help="by default True, use lider, else camera.")
    parser.add_argument('--lidar_range', type=float, default=20, help="meter, the lidar range for exploration")
    parser.add_argument('--max_a', type=float, default=2, help="the max acceleration")
    parser.add_argument('--max_sa', type=float, default=np.pi / 3, help="the max Steering Angle")
    
    """
    navigation method config
    """
    parser.add_argument('--collision_threshold', type=float, default = 0.75, help="the parameter in DWA")
    parser.add_argument('--vel_resolution', type=int, default = 3, help="the number of choice of velocity in DWA")
    parser.add_argument('--delta_resolution', type=int, default = 25, help="the number of choice of angle in DWA")
    parser.add_argument('--local_step', type=int, default = 40, help="the number of local step in pipeline")
    parser.add_argument('--multi_para', type=float, default = 3, help="the multi parameter in DWA")
    args = parser.parse_args()
    if args.yaml_file is not None:
        with open(args.yaml_file, 'r') as file:
            yaml_config = yaml.safe_load(file)
            for key, value in yaml_config.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(args, sub_key):
                            setattr(args, sub_key, sub_value)
                else:
                    if hasattr(args, key):
                        setattr(args, key, value)
    args = vars(args)
    return args

def add_ft_config(config):
    if config['algo'] == 'ft_rrt' and 'maze' in config['yaml_file']:
        config['ft_para'] = {
            'clear_radius': 10,
            'cluster_radius': 3.0,
            'utility_radius': 10, 
            'random_goal': True,
            'expand_dis': 20.0
        }
    elif config['algo'] == 'ft_rrt' and 'random' in config['yaml_file']:
        config['ft_para'] = {
            'clear_radius': 10,
            'cluster_radius': 10.0,
            'utility_radius': 10, 
            'random_goal': True,
            'expand_dis': 30.0
        }
    elif config['algo'] == 'ft_voronoi' and 'maze' in config['yaml_file']:
        config['ft_para'] = {
            'clear_radius': 3,
            'cluster_radius': 15.0,
            'utility_radius': 10, 
            'random_goal':True
        }
    elif config['algo'] == 'ft_voronoi' and 'random' in config['yaml_file']:
        config['ft_para'] = {
            'clear_radius': 3,
            'cluster_radius': 3.0,
            'utility_radius': 10, 
            'random_goal':True
        }
    elif config['algo'] == 'ft_apf' and 'maze' in config['yaml_file']:
        config['ft_para'] = {
            'clear_disk': False,
            'random_goal': True,
            'cluster_radius': 5.0,
            'k_attract': 1.0, 
            'k_agents': 1.0, 
            'AGENT_INFERENCE_RADIUS': 20.0, 
            'num_iters': 1000, 
            'repeat_penalty': 5.0, 
            'dis_type': 'l1', 
            'use_random': True, 
            'clear_radius': 15
        }
    elif config['algo'] == 'ft_apf' and 'random' in config['yaml_file']:
        config['ft_para'] = {
            'clear_disk': True,
            'random_goal': True,
            'cluster_radius': 10.0,
            'k_attract': 5.0, 
            'k_agents': 10.0, 
            'AGENT_INFERENCE_RADIUS': 20.0, 
            'num_iters': 1000, 
            'repeat_penalty': 15.0, 
            'dis_type': 'l1', 
            'use_random': True, 
            'clear_radius': 10
        }
    else:
        assert False, "Deployment has not been implemented yet."
    return config

def make_env(
        scene_params: dict,
        force_coop: bool = False,
        **env_params):
    """
    construct the environment and register.
    Args:
        :param environment_name: name of the environment
        :param map_name: name of the scenario
        :param force_coop: enforce the reward return of the environment to be global
        :param env_params: parameters that can be pass to the environment for customizing the environment

    Returns:
        Tuple[MultiAgentEnv, Dict]: env instance & env configuration dict
    """

    # default config
    env_config_file_path = './yaml/maexp.yaml'

    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = marl.dict_update(env_config_dict["env_args"], env_params, True)

    # user commandline config
    user_env_args = {}
    for param in marl.SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = marl.dict_update(env_config_dict["env_args"], user_env_args, True)
    env_config_dict["force_coop"] = force_coop

    # combine with exp and scene running config
    env_config = marl.set_ray(env_config_dict)
    for key, value in scene_params.items():
        env_config['env_args'][key] = value
    # initialize env
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                env_config["env"]))

    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]

    if env_config["force_coop"]:
        register_env(env_reg_name, lambda _: COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    else:
        register_env(env_reg_name, lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config



def make_date(save_path, index, obs_list, grid_map_list, cumulative_rewards, state_list, goal_list, type_ = 'pt'):
    # 使用npy储存数据。
    min_reward = -100 # 0.1
    if type_ == 'npy': 
        steps, agents = cumulative_rewards.shape
        IDs = np.arange(0, agents, dtype=np.int32)
        npz_files = glob.glob(os.path.join(save_path, '*.npz'))
        npz_files = sorted(npz_files, key=lambda s: int(s.split('/')[-1].split('.')[0])) 
        if len(npz_files) == 0:
            num = 0
        else:
            num = int(npz_files[-1].split('/')[-1].split('.')[0])
        for i in range(steps):
            for j in range(agents):
                if cumulative_rewards[i, j] > min_reward:
                    num += 1
                    own_image = obs_list[i][j, :3].reshape(1, 3, 128, 128) # 3，128，128
                    GT = obs_list[i][j, 3:]
                    ID_ = np.concatenate((IDs[:j], IDs[j+1:]))
                    nei_image = obs_list[i][ID_ , :3] # 前三个是mask的图 (n-1,3,128,128)
                    input_ = np.concatenate((own_image, nei_image), axis=0)
                    own_state = state_list[i][j].reshape(1, 4)
                    nei_state = state_list[i][ID_]
                    state = np.concatenate((own_state, nei_state), axis=0)
                    goal = goal_list[i][j]
                    state = rearrange(state, 'n l -> (n l)' )
                    labal = np.concatenate((state, goal), axis=0)
                    np.savez(os.path.join(save_path, f'{num}.npz'), input=input_, labal=labal, GT=GT)
    elif type_ == 'pt':
        steps, agents = cumulative_rewards.shape
        IDs = np.arange(0, agents, dtype=np.int32)
        pt_files = glob.glob(os.path.join(save_path, 'ours', str(index), '*.pt'))
        pt_files = sorted(pt_files, key=lambda s: int(s.split('/')[-1].split('.')[0]))
        num = int(pt_files[-1].split('/')[-1].split('.')[0]) if pt_files else 0

        for i in range(steps):
            for j in range(agents):
                if cumulative_rewards[i, j] > min_reward:
                    num += 1
                    own_image = obs_list[i][j]
                    ID_ = np.concatenate((IDs[:j], IDs[j+1:]))
                    # Ours
                    Ours_own = own_image['Ours'][:3].reshape(1, 3, 128, 128)
                    Ours_GT = own_image['Ours'][3:]
                    MAexp_own = own_image['maexp_input'].reshape(1, 4, 128, 128)
                    MAexp_GT = own_image['maexp_gt']
                    maans_own = own_image['maans_input'].reshape(1, 6, 128, 128)
                    maans_GT = own_image['maans_gt']


                    Ours_nei = []
                    Maexp_nei = []
                    maans_nei = []
                    for idx in ID_:
                        Ours_nei.append(obs_list[i][idx]['Ours'][:3])
                        Maexp_nei.append(obs_list[i][idx]['maexp_input'])
                        maans_nei.append(obs_list[i][idx]['maans_input'])

                    Ours_nei = np.array(Ours_nei)
                    Ours_input_data = np.concatenate((Ours_own, Ours_nei), axis = 0)
                    Maexp_nei = np.array(Maexp_nei)
                    Maexp_input_data = np.concatenate((MAexp_own, Maexp_nei), axis = 0)
                    maans_nei = np.array(maans_nei)
                    maans_input_data = np.concatenate((maans_own, maans_nei), axis = 0)
                    
                    if i+1 < steps:
                        next_state = state_list[i+1][j][:2]
                    else:
                        next_state = goal_list[i][j].astype(np.float32) 

                    own_state = state_list[i][j].reshape(1, 4)
                    nei_state = state_list[i][ID_]
                    state = np.concatenate((own_state, nei_state), axis=0)
                    goal = goal_list[i][j]
                    state = rearrange(state, 'n l -> (n l)')
                    label = np.concatenate((state, goal), axis=0)
                    grid_map = grid_map_list[i][j]
                    # own_image = obs_list[i][j, :3].reshape(1, 3, 128, 128)  # 3, 128, 128
                    # GT = obs_list[i][j, 3:]
                    # ID_ = np.concatenate((IDs[:j], IDs[j+1:]))
                    # nei_image = obs_list[i][ID_, :3]  # 前三个是mask的图 (n-1, 3, 128, 128)
                    # input_data = np.concatenate((own_image, nei_image), axis=0)
                    
                    

                    # 转换为 PyTorch 张量并保存
                    ID_ = np.concatenate((np.expand_dims(np.array(j), axis = 0), ID_))
                    IDs_ = torch.from_numpy(ID_).int()
                    Ours_input_data = torch.from_numpy(Ours_input_data).float()
                    Maexp_input_data = torch.from_numpy(Maexp_input_data).float()
                    maans_input_data = torch.from_numpy(maans_input_data).float()
                    label_tensor = torch.from_numpy(label).float()
                    Ours_GT = torch.from_numpy(Ours_GT).float()
                    MAexp_GT = torch.from_numpy(MAexp_GT).float()
                    maans_GT = torch.from_numpy(maans_GT).float()
                    grid_map = torch.from_numpy(grid_map).float()
                    
                    
                    return_ = torch.Tensor([cumulative_rewards[i, j]]).float()

                    torch.save({'input': Ours_input_data, 'label': label_tensor,
                                'GT': Ours_GT, 'Return': return_,
                                'next_state': next_state
                                }, os.path.join(save_path, 'ours', str(index), f'{num}.pt'))
                    
                    # torch.save({'label': label_tensor,
                    #             'maans_input': maans_input_data, 'maans_GT': maans_GT,
                    #             'grid_map': grid_map,
                    #             'ID': IDs_
                    #             }, os.path.join(save_path, 'maans', str(index), f'{num}.pt'))
                    
                    # torch.save({'label': label_tensor,
                    #             'Maexp_input': Maexp_input_data, 'Maexp_GT': MAexp_GT,
                    #             'grid_map': grid_map,
                    #             'ID': IDs_
                    #             }, os.path.join(save_path, 'maexp', str(index), f'{num}.pt'))


def calculate_count_reward(rewards, gamma = 0.6):
    rewards_array = np.array(rewards)
    n_steps, n_agents = rewards_array.shape
    discounted_rewards = np.zeros_like(rewards_array, dtype=float)
    for agent in range(n_agents):
        temp_discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards_array[:, agent]):
            cumulative_reward = reward + gamma * cumulative_reward
            temp_discounted_rewards.append(cumulative_reward)
        discounted_rewards[:, agent] = temp_discounted_rewards[::-1]

    return discounted_rewards

if __name__ == "__main__":
    env_config = get_config()
    env_config = add_ft_config(env_config)
    env = Multiagent_exploration(env_config)
    env.reset()
    obs_list = []
    reward_list = []
    state_list = []
    goal_list = []
    grid_map_list = []
    make_data = env_config['make_data']
    if make_data:
        save_path = './test_make_data'
        index = 12
        print('data save in'+ save_path + '/' + str(index))
        folders = ['ours'] # ['ours', 'maans', 'maexp']
        for folder in folders:
            folder_path = os.path.join(save_path, folder, str(index))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    for i in range(9000000):
        obs, final_rewards, done, goal = env.step()
        reward_list.append(np.array(list(final_rewards.values())))
        if (env.infos['local_Steps']-1) % env_config['local_step'] == 0: 
            all_obs = []
            all_grid_map = []
            all_state_list = []
            for i, name in enumerate(obs):
                all_obs.append(obs[name]['obs'])
                all_grid_map.append(obs[name]['grid_map'])
                all_state_list.append(obs[name]['all_agent_last_state'])
            obs_list.append(all_obs)
            grid_map_list.append(all_grid_map)
            goal_list.append(goal.cpu().numpy())
            state_list.append(np.array(all_state_list))
        if done['__all__'] == True or env.infos['local_Steps'] == env_config['local_step']*env_config['max_global_step']:
            if env.infos['Explore_ratio'] > 0.55:
                # merge reward
                reward_list[0] = np.zeros((env_config['num_agent']))
                cumulative_rewards = []
                for i in range(0, len(reward_list), env_config['local_step']):
                    sum_reward = np.sum(reward_list[i:i+env_config['local_step']], axis=0)
                    cumulative_rewards.append(sum_reward)
                cumulative_rewards = calculate_count_reward(cumulative_rewards)
                if make_data:
                    make_date(save_path, index, obs_list, grid_map_list, cumulative_rewards, state_list, goal_list)
            env.reset()
            obs_list = []
            reward_list = []
            state_list = []
            goal_list = []