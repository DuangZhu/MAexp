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
from env_utils.agent_v5 import Agent_explorer
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from bulid_my_model_v2 import build_model
from einops import rearrange
import os
from PIL import Image
import argparse
from collections import deque
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
            'obs': Box(low=-1000., high=1000., shape=(3, 128, 128), dtype=np.float32),
            'state_': Box(low=-1000., high=1000., shape=(3,4), dtype=np.float32),
            'others_obs': Box(low=-1000., high=1000., shape=(2, 64, 128), dtype=np.float32),
            'IDs': Box(low=0, high=10, shape=(3,), dtype=np.dtype('int64')),
            })
        # Planning Config
        self.vel_range = torch.linspace(-self.config['max_a'], self.config['max_a'], self.config['vel_resolution'], device = self.config['device'])
        self.delta_range = torch.linspace(-self.config['max_sa'], self.config['max_sa'], self.config['delta_resolution'], device = self.config['device'])
        grid1, grid2 = torch.meshgrid(self.vel_range, self.delta_range)
        self.DWA_choice = torch.stack((grid1.reshape(-1), grid2.reshape(-1)), dim=1)
        self.DWA_choice = self.DWA_choice.unsqueeze(0).repeat(self.config['num_agent'],1,1)
        # for curricula
        self.curricula_level = 11
        self.curricula_deque = deque([0]*50, maxlen = 50)
        self.curricula_deque_reward = deque([-100]*50, maxlen = 50)
        self.reward_low = -2.0
        self.agent_init_state = []
        self.max_level = 11
        self.message_lost_ratio = 0.0
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
            self.textset = torch.load(self.config['testset_path'])
            self.textset['test_id'] = self.textset['test_id'][:100]
            self.textset['text_set'] = self.textset['text_set'][:100]

    def reset(self, encoder, init_pos = None):
        print(self.infos)
        print(self.agent_init_state)
        if self.save_process:
            self.Process_data = {'Trajectory':[], 'explored_area':[]}
        if not self.config['is_train']:
            map_num = self.textset['test_id'][self.infos['Episode']].int()
        else:
            map_num = np.random.randint(0, self.config['training_map_num'])
        self.load_map(map_count=map_num) 
        print("map_ID:", self.config['scene'], self.config['map_list'][map_num])
        self.dw = self.map_w.cpu()/self.config['region']/2
        self.dh = self.map_h.cpu()/self.config['region']/2
        self.explored_space = torch.zeros((1, 3), device=self.config['device'])
        self.scene_points = {
            'map_obstacles': self.map_obstacles,
            'map_freespace': self.map_freespace,
            'map_boundary': self.map_boundary,
            'map_real_w': self.map_w,
            'map_real_h': self.map_h
        }
        self.all_area = len(self.map_freespace)
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
        else: # course training
            self.curricula_deque.append(self.infos['Explore_ratio'])
            self.curricula_deque_reward.append(self.infos['Total Reward'])
            avg_coverage = np.mean(list(self.curricula_deque))
            avg_reward = np.mean(list(self.curricula_deque_reward))
            if (avg_coverage > 0.85 or avg_reward > self.reward_low) and self.curricula_level < self.max_level: 
                # level up
                self.curricula_level += 1
                self.reward_low += 1.5
                print('The level is up to',str(self.curricula_level))
                self.curricula_deque = deque([0]*50, maxlen = 50)
                self.curricula_deque_reward = deque([-100]*50, maxlen = 50)
            print('The average coverage ratio is', str(avg_coverage))
            print('The average reward is', str(avg_reward))
            # generate the max init distance
            max_init_dis = self.scene_points['map_real_w']/6 + self.curricula_level * 10 

        self.goal_from_Gmap = False # the action from local feature map or global(G) feature map
        for key, value in self.scene_points.items():
            self.config[key] = value
        # Reward normalization based on the algorithm used
        if self.config['algo'] in [ 'ippo','vdppo','vda2c']:
            self.reward_norm = 6000 / self.all_area * 0.01
        elif self.config['algo'] in ['matrpo', 'itrpo','mappo']:
            self.reward_norm = 6000 / self.all_area * 0.01
        obs = {}
        self.agents_m = {}
        self.reward_list = {}
        self.path_len = torch.zeros(self.config['num_agent'], device = self.config['device'])
        self.explore_merged_map = torch.zeros((1, 3), device=self.config['device']) 
        '''
        init agent randomly
        '''
        self.env_vision = None
        if init_pos is not None:
            self.agent_state = init_pos.to(self.config['device'])
            self.is_collision = torch.zeros(self.config['num_agent'], device = self.config['device'])
            for i in range(self.config['num_agent']):
                agent_ = Agent_explorer(i, self.agent_state[i], self.config) # type: ignore
                self.agents_m[self.agents[i]] = agent_
                self.reward_list[self.agents[i]] = {'explore_reward':[], 'overlap_reward':[], 'total_reward':[], 'time_reward':[]}
            self.env_vision = self.collect_agent_state()
            self.detect_collision(init = True)
        elif self.config['is_train']:
            ready = False
            mask = ((self.map_freespace[:, 0] < max_init_dis) & (self.map_freespace[:, 1] < max_init_dis))
            select_init = self.map_freespace[mask]
            while ready == False:
                self.agent_state = torch.zeros([self.config['num_agent'], 4], device=self.config['device'])
                self.agent_state[:,:2] = select_init[torch.randperm(len(select_init))[:self.config['num_agent']]][:, :2]
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
                    self.agent_init_state = self.agent_state
                    
        else:
            self.agent_state = self.textset['text_set'][self.infos['Episode']].to(self.config['device'])
            self.is_collision = torch.zeros(self.config['num_agent'], device = self.config['device'])
            for i in range(self.config['num_agent']):
                agent_ = Agent_explorer(i, self.agent_state[i], self.config) # type: ignore
                self.agents_m[self.agents[i]] = agent_
                self.reward_list[self.agents[i]] = {'explore_reward':[], 'overlap_reward':[], 'total_reward':[], 'time_reward':[]}
            self.env_vision = self.collect_agent_state()
            self.detect_collision(init = True)
            print(self.agent_state)

        self.infos['Episode'] += 1
        self.infos['Steps'] = 0
        self.infos['local_Steps'] = 0
        self.infos['map_id'] = map_num
        positions = self.env_vision['all_agent_state']
        distances = torch.cdist(positions, positions)
        distances.fill_diagonal_(float('inf'))
        closest_indices = torch.topk(distances, k=2, largest=False, dim=1).indices
        
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            agent.update_agent(self.env_vision) 
            ID_ = np.concatenate(([i], closest_indices[i].cpu().numpy()))
            obs[self.agents[i]] = {"obs": agent.get_observation(ID_)[3:].cpu().numpy(), "state_": self.env_vision['all_agent_state'][ID_].cpu().numpy(),
                                    "IDs": ID_,
                                    }
        self.env_vision = self.collect_agent_state()
        others_obs = self.env_vision['all_agent_map'].to(next(encoder.parameters()).device)
        others_obs = encoder(others_obs).detach()
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            ID_ = closest_indices[i] 
            if len(self.agents_m)>2:
                obs[self.agents[i]]["others_obs"] = others_obs[ID_].cpu().numpy()
                for id in ID_:
                    agent.previous_neb_obs[str(int(id.cpu()))] = others_obs[id].cpu().numpy()
            else:
                obs[self.agents[i]]["others_obs"] = others_obs[ID_[0]].expand(2, -1, -1).cpu().numpy()
                agent.previous_neb_obs[str(int(ID_[0].cpu()))] = others_obs[ID_[0]].cpu().numpy()
            
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
        if not self.config['is_train']:
            """
            Render Configuration
            """
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.car_model = [o3d.geometry.PointCloud() for i in range(self.config['num_agent'])]
            self.vis_bound = [o3d.geometry.PointCloud() for i in range(self.config['num_agent'])]
            self.goal_point_cloud = o3d.geometry.PointCloud()
            for geometry in self.car_model:
                self.vis.add_geometry(geometry)
            for geometry in self.vis_bound:
                self.vis.add_geometry(geometry)
            
            map_obstacles_o3d = o3d.geometry.PointCloud()
            map_obstacles_o3d.points = o3d.utility.Vector3dVector(self.map_obstacles.cpu())
            map_obstacles_o3d.paint_uniform_color([0.6, 0.6, 0.6])
            self.vis.add_geometry(map_obstacles_o3d)
            
            self.explored_space_o3d = o3d.geometry.PointCloud()
            self.explored_space_o3d.points = o3d.utility.Vector3dVector(self.map_freespace.cpu())
            self.vis.add_geometry(self.explored_space_o3d)
            
            self.map_freespace_o3d = o3d.geometry.PointCloud()
            self.map_freespace_o3d.points = o3d.utility.Vector3dVector(self.map_freespace.cpu())
            self.vis.add_geometry(self.map_freespace_o3d)
        return obs
    
    def step(self, action, encoder):
        self.infos['Steps'] += 1
        final_rewards = {}
        sum_rewards = np.zeros(self.config['num_agent'])
        obs = {}
        info = {}
        done = {'__all__': False}
        region, goal = self.get_goal(action)
        agents_bound_map = torch.zeros((self.config['num_agent'], 8000, 2), device=self.config['device'])
        for i, agent in enumerate(self.agents_m.values()):
            # agent = self.agents_m[name]
            agents_bound_map[i,:len(agent.detected_bound_for_map)] = agent.detected_bound_for_map[:,:2]
        for step in range(self.config['local_step']):
            self.infos['local_Steps'] += 1
            _, action = self.DWA(goal, agents_bound_map)
            for i, agent in enumerate(self.agents_m.values()):
                agent.update_agent(self.env_vision)
                agent.step(action[i]) 
            positions = self.env_vision['all_agent_state']
            distances = torch.cdist(positions, positions)
            distances.fill_diagonal_(float('inf'))
            closest_indices = torch.topk(distances, k=2, largest=False, dim=1).indices
            for i, name in enumerate(self.agents_m):
                agent = self.agents_m[name]
                ID_ = np.concatenate(([i], closest_indices[i].cpu().numpy()))
                obs[name] = {"obs": agent.get_observation(ID_)}
                agents_bound_map[i,:len(agent.detected_bound_for_map)] = agent.detected_bound_for_map[:,:2]
                
            rewards = self.calculate_reward()
            sum_rewards += rewards
            # Visualzation
            if self.save_process:
                self.Process_data['Trajectory'].append(self.env_vision['all_agent_state'])
            if not self.config['is_train']:
                self.add_Goal(goal)
                self.render() 
            self.env_vision = self.collect_agent_state()
            self.detect_collision()
            del_l = self.env_vision['all_agent_state'][:,:2] - self.env_vision['all_agent_last_state'][:,:2]
            self.path_len += torch.sqrt(torch.sum(del_l**2, dim=1))
            if not self.threshold_reached_85 and self.infos['Explore_ratio'] > 0.85:
                print('reach_85%_coverage')
                self.threshold_reached_85 = True
                self.infos['85% Coverage'] = self.infos['local_Steps']
                maps = self.env_vision['all_agent_global_map'][:,0]
                merge_map = torch.sum(maps, dim = 0)
                area_a = torch.nonzero(merge_map).size(0)
                area_b = (merge_map > 0.6).sum().item()
                self.infos['85% Overlap'] = area_b/area_a
                self.infos['85% pathlen'] = float(torch.sum(self.path_len).cpu())
            if self.infos['Explore_ratio'] > 0.95:
                done = {key: True for key in done}
                sum_rewards += np.ones(self.config['num_agent'])  * 100 *  self.reward_norm
                break
            elif all(self.env_vision['all_agent_collision']) or self.infos['Steps'] > self.config['max_global_step']:
                if all(self.env_vision['all_agent_collision']):
                    print('Destroy!!!')
                    sum_rewards -= np.ones(self.config['num_agent']) * 200 * self.reward_norm
                done = {key: True for key in done}
                break

        positions = self.env_vision['all_agent_state']
        distances = torch.cdist(positions, positions)
        distances.fill_diagonal_(float('inf'))
        closest_indices = torch.topk(distances, k=2, largest=False, dim=1).indices
        self.infos['Agents Rewards'] += sum_rewards
        others_obs = self.env_vision['all_agent_map'].to(next(encoder.parameters()).device)
        others_obs = encoder(others_obs).detach()
        for i, name in enumerate(self.agents_m):
            final_rewards[name] = sum_rewards[i]
            info[name] = {}
            done[name] = False
            ID_ = np.concatenate(([i], closest_indices[i].cpu().numpy()))
            obs[name]['obs'] = obs[name]['obs'][3:].cpu().numpy()
            obs[name]["state_"] = self.env_vision['all_agent_state'][ID_].cpu().numpy()
            obs[name]["others_obs"] = others_obs[ID_[1:]].cpu().numpy()
            if self.message_lost_ratio > 0:
                mask = np.random.rand(self.config['num_agent']-1) <= self.message_lost_ratio
                if len(self.agents_m)>2:
                    for j, id in enumerate(ID_[1:]):
                        if mask[j]:
                            if str(id) in self.agents_m[name].previous_neb_obs:
                                obs[name]["others_obs"][j] = self.agents_m[name].previous_neb_obs[str(id)]
                            else:
                                available_values = list(self.agents_m[name].previous_neb_obs.values())
                                obs[name]["others_obs"][j] = random.choice(available_values)
                        else:
                            self.agents_m[name].previous_neb_obs[str(id)] = obs[name]["others_obs"][j]       
                else:
                    if mask[0]:
                        obs[name]["others_obs"] = np.expand_dims(self.agents_m[name].previous_neb_obs[str(int(closest_indices[i, 0]))], axis=0)
                        obs[name]["others_obs"] = np.repeat(obs[name]["others_obs"], 2, axis=0)
                    else:
                        self.agents_m[name].previous_neb_obs[str(int(closest_indices[i,0].cpu()))] = obs[name]["others_obs"][0]
            obs[name]["IDs"] = ID_
            
        if done['__all__'] == True:
            if self.infos['Explore_ratio'] > 0.95:
                self.calculate_final_metric(is_95 = True)
                self.infos['95% pathlen'] = float(torch.sum(self.path_len).cpu())
            else:
                self.calculate_final_metric(is_95 = False)
        return obs, final_rewards, done, info
    

    def get_goal(self, actions):
        region = torch.zeros(len(actions))
        location = torch.zeros((len(actions), 2))
        for i, name in enumerate(actions):
            if self.goal_from_Gmap:
                region[i] = actions[name][0]
                location[i] = torch.tensor(actions[name][1])
            else:
                region[i] = actions[name][0]
                location[i] = torch.tensor(actions[name][1])
        y = region % self.config['region']
        x = torch.div(region, self.config['region'], rounding_mode='floor')
        x = (2 * x + 1 - location[:, 0]) * self.dw
        y = (2 * y + 1 - location[:, 1]) * self.dh
        goals = torch.stack((x, y), dim=1).to(self.config['device'])
        if not self.goal_from_Gmap:
            goals = self.local_to_global(goals, self.env_vision['all_agent_state'])
        return region, goals
    
    def local_to_global(self, local_point, car_pos):
        '''
        Calcluate the global position of the goal from local map
        '''
        center = torch.tensor([self.map_w / 2, self.map_h / 2])
        x = local_point[:, 0] - center[0]
        y = local_point[:, 1] - center[1]
        return torch.stack((car_pos[:, 0] + x, car_pos[:, 1] + y), dim=1).to(self.config['device'])
    
    def calculate_final_metric(self, is_95):
        if is_95:
            print('reach_95%_coverage')
            self.infos['95% Coverage'] = self.infos['local_Steps']
            maps = self.env_vision['all_agent_global_map'][:,0]
            merge_map = torch.sum(maps, dim = 0)
            area_a = torch.nonzero(merge_map).size(0)
            area_b = (merge_map > 0.6).sum().item()
            self.infos['95% Overlap'] = area_b/area_a
            self.infos['Total Reward']  = np.sum(self.infos['Agents Rewards'])
        self.infos['Total Reward']  = np.sum(self.infos['Agents Rewards'])
        self.infos['Rewards Variance'] = np.sqrt(np.var(self.infos['Agents Rewards']))
        if self.save_process:
            self.Process_data['explored_area'].append(self.explore_merged_map)
            np.save('/remote-home/ums_zhushaohao/new/2024/visual_data/scal/random/4/'+str(self.infos['Episode'])+'.npy', self.Process_data)
        
    
    def calculate_reward(self):
        rewards = np.zeros(self.config['num_agent'])
        """
        overlap penalty
        """
        overlap = torch.zeros(self.config['num_agent'], self.config['num_agent'], device = self.config['device'], dtype=torch.int)
        for i in range(self.config['num_agent']):
            for j in range(i+1, self.config['num_agent']):
                cars_overlap = torch.cat((self.agents_m[self.agents[i]].freespace_selected_points_mat,
                                           self.agents_m[self.agents[j]].freespace_selected_points_mat), dim=0)
                overlap[i,j] = cars_overlap.size()[0] - torch.unique(cars_overlap, dim = 0, sorted = False).size()[0] 
        overlap_reward = (overlap.T + overlap).sum(0)

        """
        exploration reward
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
            rewards[i] = float(self.reward_list[name]['time_reward'][-1] + self.reward_list[name]['explore_reward'][-1] * 0.3 - self.reward_list[name]['overlap_reward'][-1] * 0.01)  * self.reward_norm
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
        collect state
        """
        all_agent_state = []
        all_agent_model_mats = []
        all_agent_map = []
        all_agent_global_map = []
        if self.env_vision is not None:
            all_agent_last_state = self.env_vision['all_agent_state'].clone()
        else:
            all_agent_last_state = None
        for name in self.agents_m:
            agent = self.agents_m[name]
            all_agent_state.append(agent.agent_state)
            all_agent_model_mats.append(agent.car_model_mat)
            all_agent_map.append(agent.img[3:])
            all_agent_global_map.append(agent.obs)
            self.explored_space = torch.cat((self.explored_space, agent.explored_space), dim=0) # 储存了merge map,是所有人共同探索的区域
            self.explored_space = torch.unique(self.explored_space, dim = 0, sorted = False)
        all_agent_state = torch.cat(all_agent_state, dim=0).reshape(-1,4)
        all_agent_model_mats = torch.cat(all_agent_model_mats, dim=0)
        all_agent_map = torch.stack(all_agent_map)
        all_agent_global_map = torch.stack(all_agent_global_map)
        env_vision = {
            'all_agent_state': all_agent_state, 
            'all_agent_model_mats': all_agent_model_mats,
            'all_agent_collision': self.is_collision,
            'all_agent_map': all_agent_map,
            'all_agent_global_map': all_agent_global_map,
            'all_agent_last_state': all_agent_last_state
        }
        return env_vision
    
    def detect_collision(self, init = False):
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
        circle_points = []
        for point in points:
            theta1 = np.linspace(0, 2 * np.pi, 100)
            circle_x = point[0] + np.cos(theta1)*0.2*self.config['map_resolution']
            circle_y = point[1] + np.sin(theta1)*0.2*self.config['map_resolution']
            circle_z = np.zeros_like(theta1)
            circle_points.extend(np.column_stack((circle_x, circle_y, circle_z)))

        circle_points = np.array(circle_points)
        points_3d = circle_points
        
        self.goal_point_cloud.points = o3d.utility.Vector3dVector(points_3d + np.array([0, 0, 0.03]))
        colors = np.zeros((len(points_3d), 3)) 


        colors[:len(points_3d)//3] = [245/255, 108/255, 108/255]
        colors[len(points_3d)//3:len(points_3d)//3*2] = [253/255, 210/255, 224/255]
        colors[len(points_3d)//3*2:] = [245/255, 150/255, 125/255]
        self.goal_point_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.goal_point_cloud)     

        
    def render(self):
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
            self.vis_bound[i].paint_uniform_color([0.6, 0.6, 0.6]) 
            self.vis.update_geometry(self.car_model[i])
            self.vis.update_geometry(self.vis_bound[i])
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.capture:
            save_path = '/home/shaohao/Documents/MAexp/img'
            subfolders = ['all', 'agent_0', 'agent_1', 'agent_2']
            if not os.path.exists(save_path):
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
        self.vis.capture_screen_image(filename)
        print(f"Saved screenshot: {filename}")   

  
    def load_map(self, map_count = None, id = None):
        """
        load pointcloud map
        """
        if map_count is not None:
            self.map_freespace = torch.from_numpy(np.load("./map/"+self.config['scene']+"/"+self.config['map_list'][map_count]+"_freespace.npy")).float().to(self.config['device'])
            self.map_obstacles = torch.from_numpy(np.load("./map/"+self.config['scene']+"/"+self.config['map_list'][map_count]+"_obstacles.npy")).float().to(self.config['device'])
            self.map_boundary = torch.from_numpy(np.load("./map/"+self.config['scene']+"/"+self.config['map_list'][map_count]+"_boundary.npy")).float().to(self.config['device'])
            if self.config['scene'] in ['maze', 'random', 'maze9', 'random2','maze_4_change', 'random3']:
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
                self.map_obstacles = torch.from_numpy(np.load("./map/"+self.config['scene']+"/map"+str(id)+"_obstacles.npy")).float().to(self.config['device'])
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
    parser.add_argument("--yaml_file", type=str, default = '/remote-home/ums_zhushaohao/new/2024/MAexp/yaml/maze.yaml', help="the yaml file for the experiment parameter")
    parser.add_argument("--result_file", type=str, default = '/remote-home/ums_zhushaohao/new/2024/MAexp/paper3_result/a.json', help="the yaml file for the experiment parameter")
    parser.add_argument("--testset_path", type=str, default = '/remote-home/ums_zhushaohao/new/2025/MAexp/testset/Final_testdata_mazes_test.pt', help="path to your testset")
    
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


if __name__ == '__main__':
    setup_seed(4)
    env_config = get_config()
    ENV_REGISTRY["maexp"] = Multiagent_exploration
    COOP_ENV_REGISTRY["maexp"] = Multiagent_exploration
    # initialize env
    env = make_env(scene_params = env_config)
    algo_class = getattr(marl.algos, env_config['algo'])
    method = algo_class(hyperparam_source="common")
    # customize model
    model = build_model(env, method, {"core_arch": "vit_crossatt"}) # "vit_crossatt", "mlp"
    # start learning
    if env_config['is_train'] == True:
        method.fit(env, model, stop={'episode_reward_mean': 200000, 'timesteps_total': 10000000}, 
                    # restore_path={'params_path': "/remote-home/ums_zhushaohao/new/2024/MAexp/exp_results/vda2c_vit_crossatt_MAexp/VDA2CTrainer_maexp_MAexp_0274b_00000_0_2024-12-14_10-05-56/params.json",  # experiment configuration
                    #         'model_path': "/remote-home/ums_zhushaohao/new/2024/MAexp/exp_results/vda2c_vit_crossatt_MAexp/VDA2CTrainer_maexp_MAexp_0274b_00000_0_2024-12-14_10-05-56/checkpoint_003600/checkpoint-3600"},
                              local_mode = False, num_workers = 4, share_policy='all', checkpoint_freq=300)
    else:
        method.fit(env, model, stop={'episode_reward_mean': 200000, 'timesteps_total': 10000000}, 
                  restore_path={'params_path': "/remote-home/ums_zhushaohao/new/2024/MAexp/exp_results/vda2c_vit_crossatt_MAexp/VDA2CTrainer_maexp_MAexp_a0c78_00000_0_2024-12-23_02-57-21/params.json",  # experiment configuration
                            'model_path': "/remote-home/ums_zhushaohao/new/2024/MAexp/exp_results/vda2c_vit_crossatt_MAexp/VDA2CTrainer_maexp_MAexp_845f5_00000_0_2024-12-19_23-32-31/checkpoint_008100/checkpoint-8100"},
                              local_mode=True, num_workers = 0, share_policy='all')




# if __name__ == "__main__":
#     """
#     You can use this to visual the environment with random work strategy.
#     """

#     def generate_random_action(config):
#         action = {}
#         for i in range(config['num_agent']):
#             agent_key = f"agent_{i}"
#             random_integer = random.randint(0, 63)
#             random_floats = [round(random.uniform(-1, 1), 2) for _ in range(2)]
#             action[agent_key] = (random_integer, random_floats)
#         return action
    
#     np.random.seed(2)
#     env_config = get_config()
#     env = Multiagent_exploration(env_config)
#     logging.info("Environment created successfully!")
#     env.reset()
#     t_start = time.time()
#     for i in range(20):
#         action = generate_random_action(env_config)
#         env.step(action)
#     t_end = time.time()
#     print("Spend time", t_end - t_start)
