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
from env_utils.agent_v4 import Agent_explorer
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from bulid_my_model_v2 import build_model
from einops import rearrange
import os
from PIL import Image
import argparse
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
            'Total Reward': 0.,
            'Agents Rewards': np.zeros(self.config['num_agent']),
            'Rewards Variance': 0.
        }

        # Agent Configuration
        self.action_space = spaces.Tuple((
            Discrete(self.config['region']**2), 
            Box(low=np.array([-1.0,-1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)
            ))  
        self.observation_space = GymDict({
            'obs': Box(low=0, high=1, shape=(4, 125, 125), dtype=np.dtype('int32')),
            'state_': Box(low=-100, high=1000, shape=(self.config['num_agent'],4), dtype=np.float32),
            'others_obs': Box(low=0, high=1, shape=(self.config['num_agent']-1, 4, 125, 125), dtype=np.dtype('int32')),
            'commu': Box(low=0, high=1, shape=(self.config['num_agent']-1, ), dtype=np.dtype('bool')),
            'IDs': Box(low=0, high=self.config['num_agent'], shape=(self.config['num_agent'],), dtype=np.dtype('int32')),
            'grid_map': Box(low=0, high=self.config['region']*2, shape=(4*self.config['num_agent'], self.config['region'], self.config['region']), dtype=np.dtype('float32'))})
        self.agent_state = []

        self.load_map(self.config['training_map_num']) 
        self.map_real_w = torch.max(self.map_Obstacles[0][:,0])
        self.map_real_h = torch.max(self.map_Obstacles[0][:,1])
        
        # Planning Config
        self.vel_range = torch.linspace(-self.config['max_a'], self.config['max_a'], self.config['vel_resolution'], device = self.config['device'])
        self.delta_range = torch.linspace(-self.config['max_sa'], self.config['max_sa'], self.config['delta_resolution'], device = self.config['device'])
        grid1, grid2 = torch.meshgrid(self.vel_range, self.delta_range)
        self.DWA_choice = torch.stack((grid1.reshape(-1), grid2.reshape(-1)), dim=1)
        self.DWA_choice = self.DWA_choice.unsqueeze(0).repeat(self.config['num_agent'],1,1)

        if not self.config['is_train']:
            self.test_metrix = {'ER':[], '85C':[], '95C':[], '85O': [], '95O':[], 'RV':[]}
            self.capture = self.config['is_capture']
            
    
    def reset(self):
        """
        Reset the exploration environment.
        """
        print(self.infos)
        print(self.agent_state)
        if not self.config['is_train']:
            self.test_metrix['ER'].append(self.infos['Explore_ratio'])
            if self.infos['85% Coverage'] != 0:
                self.test_metrix['85C'].append(self.infos['85% Coverage'])
                self.test_metrix['85O'].append(self.infos['85% Overlap'])
            if self.infos['95% Coverage'] != 0:
                self.test_metrix['95C'].append(self.infos['95% Coverage'])
                self.test_metrix['95O'].append(self.infos['95% Overlap'])
            self.test_metrix['RV'].append(self.infos['Rewards Variance'])
            if self.infos['Episode'] % 100 == 0 and self.infos['Episode'] != 0:
                print('The average explore rate is ' + str(sum(self.test_metrix['ER']) / len(self.test_metrix['ER'])))
                if len(self.test_metrix['85C']) != 0:
                    print('The average 85% Coverage is ' + str(sum(self.test_metrix['85C']) / len(self.test_metrix['85C'])))
                    print('The average 85% Overlap is ' + str(sum(self.test_metrix['85O']) / len(self.test_metrix['85O'])))
                else:
                    print('The average 85% Coverage is -1')
                    print('The average 85% Overlap is -1')
                
                if len(self.test_metrix['95C']) != 0:
                    print('The average 95% Coverage is ' + str(sum(self.test_metrix['95C']) / len(self.test_metrix['95C'])))
                    print('The average 95% Overlap is ' + str(sum(self.test_metrix['95O']) / len(self.test_metrix['95O'])))
                else:
                    print('The average 95% Coverage is -1')
                    print('The average 95% Overlap is -1')
                print('The average Rewards Variance is ' + str(sum(self.test_metrix['RV']) / len(self.test_metrix['RV'])))
                with open('/data/zsh/MAexp/maexp_maze1.json', 'w') as f:
                    json.dump(self.test_metrix, f)
                sys.exit()
                 
        map_num = np.random.randint(0, self.config['training_map_num'])
        print("map_ID:", self.config['map_list'][map_num])
        self.map_obstacles = self.map_Obstacles[map_num].to(self.config['device'])
        self.map_freespace = self.map_Freespace[map_num].to(self.config['device'])
        self.map_boundary = self.map_Boundary[map_num].to(self.config['device'])
        self.all_area = len(self.map_freespace)
        self.map_w = self.map_W[map_num].to(self.config['device']) 
        self.map_h = self.map_H[map_num].to(self.config['device']) 
        self.dw = self.map_W[map_num].numpy()/self.config['region']/2
        self.dh = self.map_H[map_num].numpy()/self.config['region']/2
        self.explored_space = torch.empty((0, 3), device=self.config['device'])
        self.scene_points = {
            'map_obstacles': self.map_obstacles,
            'map_freespace': self.map_freespace,
            'map_boundary': self.map_boundary,
            'map_real_w': self.map_W[map_num].to(self.config['device']),
            'map_real_h': self.map_H[map_num].to(self.config['device'])
        }
        
        self.reward_norm = 6000 / self.all_area * 0.01 # Reward normalization
        for key, value in self.scene_points.items():
            self.config[key] = value
        obs = {}
        self.agents_m = {}
        self.reward_list = {}
        self.explore_merged_map = torch.empty((1, 3), device=self.config['device'])
        # init_car
        # self.agent_state = torch.tensor([
        #     [17, 17, 0, 0.],
        #     [17, 25, 0, 0.],
        #     [23, 20, 0, 0.]
        # ], device=self.config['device'])

        if self.agent_state != []:
            self.is_collision = torch.zeros(self.config['num_agent'], device = self.config['device'])
            for i in range(self.config['num_agent']):
                agent_ = Agent_explorer(i, self.agent_state[i], self.config) 
                self.agents_m[self.agents[i]] = agent_
                self.reward_list[self.agents[i]] = {'explore_reward':[], 'overlap_reward':[], 'total_reward':[], 'time_reward':[]}
            self.env_vision = self.collect_agent_state()
            self.detect_collision()
        elif self.agent_state == []:
            ready = False
            select_init = self.map_freespace
            while ready == False:
                self.agent_state = torch.zeros([self.config['num_agent'], 4], device=self.config['device'])
                self.agent_state[:,:2] = select_init[torch.randperm(len(select_init))[:self.config['num_agent']]][:, :2]
                self.agent_state[:,-1] = torch.rand(self.config['num_agent'], device = self.config['device']) * 2 * math.pi - math.pi
                self.is_collision = torch.zeros(self.config['num_agent'], device = self.config['device'])
                for i in range(self.config['num_agent']):
                    agent_ = Agent_explorer(i, self.agent_state[i], self.config) 
                    self.agents_m[self.agents[i]] = agent_
                    self.reward_list[self.agents[i]] = {'explore_reward':[], 'overlap_reward':[], 'total_reward':[], 'time_reward':[]}
                self.env_vision = self.collect_agent_state()
                self.detect_collision(init = True)
                if self.env_vision['all_agent_collision'].any() == False:
                    ready = True
                    self.agent_init_state = self.agent_state

        self.infos['Episode'] += 1
        self.infos['Steps'] = 0
        self.infos['local_Steps'] = 0
        self.env_vision = self.collect_agent_state()
        self.env_vision['all_agent_grid_map'] = self.calculate_grid_map()
        diff = self.env_vision['all_agent_state'][:,:2].unsqueeze(1) - self.env_vision['all_agent_state'][:,:2].unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1)) 
        dist[torch.eye(dist.size(0)).bool()] = float('nan')
        dist = dist[~torch.isnan(dist)].reshape(self.config['num_agent'], -1)  
        dist = dist < self.config['max_commu_dis']
        IDs = np.arange(0, self.config['num_agent'], dtype=np.int32)
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            agent.update_agent(self.env_vision) 
            ID_ = np.concatenate(([IDs[i]], IDs[:i], IDs[i+1:]))
            obs[self.agents[i]] = {"obs": agent.get_observation().cpu().numpy(), "state_": self.env_vision['all_agent_state'][ID_].cpu().numpy(),
                                    "others_obs": torch.cat([self.env_vision['all_agent_map'][:agent.agent_id], self.env_vision['all_agent_map'][agent.agent_id+1:]]).cpu().numpy(),
                                    "commu": dist[i].cpu().numpy(),
                                    "IDs": ID_,
                                    "grid_map": self.env_vision['all_agent_grid_map'][i].cpu().numpy()}
        self.calculate_reward()
        self.threshold_reached_85 = False
        self.infos.update({
                'Explore_ratio': 0.,
                '85% Coverage': 0.,
                '95% Coverage': 0.,
                '85% Overlap': 0.,
                '95% Overlap': 0.,
                'Total Reward': 0.,
                'Agents Rewards': np.zeros(self.config['num_agent']),
                'Rewards Variance': 0.
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
            self.vis.add_geometry(self.goal_point_cloud)

            
            # add obstacle point clouds
            map_obstacles_o3d = o3d.geometry.PointCloud()
            map_obstacles_o3d.points = o3d.utility.Vector3dVector(self.map_obstacles.cpu())
            map_obstacles_o3d.paint_uniform_color([0.6, 0.6, 0.6])
            self.vis.add_geometry(map_obstacles_o3d)
            
            # add explored space point clouds
            self.explored_space_o3d = o3d.geometry.PointCloud()
            self.explored_space_o3d.points = o3d.utility.Vector3dVector(self.map_freespace.cpu())
            self.vis.add_geometry(self.explored_space_o3d)
            
            # # add unexplored space point clouds
            # self.map_freespace_o3d = o3d.geometry.PointCloud()
            # self.map_freespace_o3d.points = o3d.utility.Vector3dVector(self.map_freespace.cpu())
            # self.vis.add_geometry(self.map_freespace_o3d)
        return obs
    
    def step(self, action): 
        """
        Advance the environment by one timestep.
        """
        self.infos['Steps'] += 1
        final_rewards = {}
        sum_rewards = np.zeros(self.config['num_agent'])
        obs = {}
        info = {}
        done = {'__all__': False}
        region, goal = self.get_goal(action)
        agents_bound_map = torch.zeros((self.config['num_agent'], 8000, 2), device=self.config['device'])
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            agents_bound_map[i,:len(agent.detected_bound_for_map)] = agent.detected_bound_for_map[:,:2]
        for step in range(self.config['local_step']):
            self.infos['local_Steps'] += 1
            _, action = self.DWA(goal, agents_bound_map)
            for i, name in enumerate(self.agents_m):
                agent = self.agents_m[name]
                agent.update_agent(self.env_vision)
                agent.step(action[i]) 
                obs[name] = {"obs": agent.get_observation()}
                agents_bound_map[i,:len(agent.detected_bound_for_map)] = agent.detected_bound_for_map[:,:2]
            rewards = self.calculate_reward()
            sum_rewards += rewards
            if not self.config['is_train']:
                self.add_Goal(goal)
                self.render() 
            self.env_vision = self.collect_agent_state()
            self.env_vision['all_agent_grid_map'] = self.calculate_grid_map(region)
            self.detect_collision()
            if not self.threshold_reached_85 and self.infos['Explore_ratio'] > 0.85:
                print('reach_85%_coverage')
                self.threshold_reached_85 = True
                self.infos['85% Coverage'] = self.infos['local_Steps']
                maps = self.env_vision['all_agent_map'][:,0]
                merge_map = torch.sum(maps, dim = 0)
                area_a = torch.nonzero(merge_map).size(0)
                area_b = (merge_map > 1).sum().item()
                self.infos['85% Overlap'] = area_b/area_a
            if self.infos['Explore_ratio'] > 0.95:
                done = {key: True for key in done}
                sum_rewards += np.ones(self.config['num_agent'])  * 100 *  self.reward_norm
                self.calculate_final_metric(is_95 = True)
                break
            elif all(self.env_vision['all_agent_collision']) or self.infos['Steps'] > self.config['max_global_step']:
                if all(self.env_vision['all_agent_collision']):
                    print('Destroy!!!')
                    sum_rewards -= np.ones(self.config['num_agent']) * 200 * self.reward_norm
                done = {key: True for key in done}
                self.calculate_final_metric(is_95 = False)
                break
        diff = self.env_vision['all_agent_state'][:,:2].unsqueeze(1) - self.env_vision['all_agent_state'][:,:2].unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1)) 
        dist[torch.eye(dist.size(0)).bool()] = float('nan')
        dist = dist[~torch.isnan(dist)].reshape(self.config['num_agent'], -1)  
        dist = dist < self.config['max_commu_dis']
        IDs = np.arange(0, self.config['num_agent'], dtype=np.int32)
        self.infos['Agents Rewards'] += sum_rewards
        for i, name in enumerate(self.agents_m):
            final_rewards[name] = sum_rewards[i]
            info[name] = {}
            done[name] = False
            ID_ = np.concatenate(([IDs[i]], IDs[:i], IDs[i+1:]))
            obs[name]['obs'] = obs[name]['obs'].cpu().numpy()
            obs[name]["state_"] = self.env_vision['all_agent_state'][ID_].cpu().numpy()
            obs[name]["others_obs"] = torch.cat([self.env_vision['all_agent_map'][:agent.agent_id], self.env_vision['all_agent_map'][agent.agent_id+1:]]).cpu().numpy()
            obs[name]["commu"] = dist[i].cpu().numpy()
            obs[name]["IDs"] = ID_
            obs[name]["grid_map"] = self.env_vision['all_agent_grid_map'][i].cpu().numpy()
        return obs, final_rewards, done, info
    
    def calculate_grid_map(self, action = None):
        """
        calculate the grid maps, which are parts of the observation, for agents.
        """
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
        out_pos[:,1:2*self.config['num_agent']:2,:,:] = ypos.unsqueeze(-2).expand(*ypos.shape[:-1], self.config['region'], self.config['region'])
        out_pos += torch.ones_like(out_pos, device=self.config['device'])*(self.config['region'] - 1) #(0-14)
        if action is None:
            out_goal = out_pos
        else:
            region = action.to(self.config['device'])
            x = region % self.config['region'] 
            y = region // self.config['region']
            region = torch.stack((x,y), dim = 1).unsqueeze(-1).expand(-1,-1,self.config['region'])
            region = I-region
            xpos = region[:,0,:].unsqueeze(0).expand(self.config['num_agent'], -1, -1) # self.num_aget, x, 8
            out_goal[:,0:2*self.config['num_agent']:2,:,:] = xpos.unsqueeze(-1).expand(*xpos.shape, self.config['region'])
            ypos = region[:,1,:].unsqueeze(0).expand(self.config['num_agent'], -1, -1) # self.num_aget, y, 8
            out_goal[:,1:2*self.config['num_agent']:2,:,:] = ypos.unsqueeze(-2).expand(*ypos.shape[:-1], self.config['region'], self.config['region'])
            out_goal += torch.ones_like(out_pos, device=self.config['device'])*(self.config['region'] - 1) #(0-14)
        out_pos_ = copy.deepcopy(out_pos)
        out_goal_ = copy.deepcopy(out_goal)
        IDs = np.arange(0, self.config['num_agent']*2, dtype=np.int32)
        for a in range(1, self.config['num_agent']):
            ID_ = np.concatenate((IDs[2*a:2*(a+1)],IDs[:2*a], IDs[2*(a+1):]))
            out_pos_[a] = out_pos[a][ID_]
            out_goal_[a] = out_goal[a][ID_]
        return torch.cat((out_pos_, out_goal_), dim = 1)
    
    
    def get_goal(self, actions):
        """
        Transforming agent actions into global goals (navigation goal).
        """
        region = torch.zeros([len(actions)])
        location = torch.zeros([len(actions), 2])
        for i, name in enumerate(actions):
            region[i] = actions[name][0]
            location[i] = torch.tensor(actions[name][1])
        y = region % self.config['region']
        x = torch.div(region, self.config['region'], rounding_mode='floor')
        x = (2*x+1-location[:,0])*self.dw
        y = (2*y+1-location[:,1])*self.dh
        x = torch.stack((x, y), dim=1).to(self.config['device'])
        return region, x

    def calculate_final_metric(self, is_95):
        """
        Calculate metric when the exploration is over.
        """
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
        
    def calculate_reward(self):
        """
        Calculate reward for Reinforcement Learning
        """
        rewards = np.zeros(self.config['num_agent'])
        """
        Overlap Penalty
        """
        overlap = torch.zeros(self.config['num_agent'], self.config['num_agent'], device = self.config['device'], dtype=torch.int)
        for i in range(self.config['num_agent']):
            for j in range(i+1, self.config['num_agent']):
                cars_overlap = torch.cat((self.agents_m[self.agents[i]].freespace_selected_points_mat,
                                           self.agents_m[self.agents[j]].freespace_selected_points_mat), dim=0)
                overlap[i,j] = cars_overlap.size()[0] - torch.unique(cars_overlap, dim = 0, sorted = False).size()[0] 
        overlap_reward = (overlap.T + overlap).sum(0)

        """
        Exploration Reward
        """
        for name in self.agents:
            map_overlap = torch.cat((self.agents_m[name].freespace_selected_points_mat,
                                           self.explore_merged_map), dim=0)
            self.reward_list[name]['explore_reward'].append(self.agents_m[name].freespace_selected_points_mat.size()[0] - map_overlap.size()[0]
                                                            + torch.unique(map_overlap, dim = 0, sorted = False).size()[0])
            self.reward_list[name]['overlap_reward'].append(overlap_reward[int(name.split('_')[-1])].cpu())
            self.reward_list[name]['time_reward'].append((-(self.all_area-self.explore_merged_map.shape[0])/self.all_area))
        # update merge map
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
        Environment collects agents' informations
        """
        all_agent_state = []
        all_agent_model_mats = []
        all_agent_map = []
        all_grid_map = []
        for name in self.agents_m:
            agent = self.agents_m[name]
            all_agent_state.append(agent.agent_state)
            all_agent_model_mats.append(agent.car_model_mat)
            all_agent_map.append(agent.img)
            self.explored_space = torch.cat((self.explored_space, agent.explored_space), dim=0)
            self.explored_space = torch.unique(self.explored_space, dim = 0, sorted = False)
        all_agent_state = torch.cat(all_agent_state, dim=0).reshape(-1,4)
        all_agent_model_mats = torch.cat(all_agent_model_mats, dim=0)
        all_agent_map = torch.stack(all_agent_map)
        env_vision = {
            'all_agent_state': all_agent_state, 
            'all_agent_model_mats': all_agent_model_mats,
            'all_agent_collision': self.is_collision,
            'all_agent_map': all_agent_map,
            'all_agent_grid_map': all_grid_map
        }
        return env_vision
    
    def detect_collision(self, init = False):
        """
        Detect collision for agents
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
        Navigtion algorithm for agents. (Could be any other navigation algorithms)
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
        target_reward = -torch.norm(torch.stack((x,y),dim=2) - target_state, dim = 2) 
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

    def add_Goal(self, goal):
        """
        Add global navigation goals when rendering
        """
        N = goal.shape[0]
        theta1 = torch.linspace(0, 2 * np.pi, 30, device=goal.device)
        goal_expanded = goal[:, None, :].expand(N, 30, 2)
        circle_x = goal_expanded[..., 0] + torch.cos(theta1) * 0.2 * self.config['map_resolution']
        circle_y = goal_expanded[..., 1] + torch.sin(theta1) * 0.2 * self.config['map_resolution']
        circle_z = torch.zeros((N, 30), device=goal.device)
        circle_points = torch.stack((circle_x, circle_y, circle_z), dim=-1)
        circle_points = circle_points.reshape(-1, 3)
        points_3d = circle_points.cpu().numpy()
        self.goal_point_cloud.points = o3d.utility.Vector3dVector(points_3d + np.array([0, 0, 0.03]))        
        colors = np.zeros((N * 30, 3))
        # colors_list = [[245/255, 108/255, 108/255],
        # [253/255, 210/255, 224/255],
        # [245/255, 150/255, 125/255]
        # ]
        # for i, color in enumerate(colors_list):
        #    colors[i * 30:(i + 1) * 30] = color
        self.goal_point_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.goal_point_cloud)
        
    def render(self):
        """
        Visualizing the exploration process when testing
        """
        self.explored_space_o3d.points = o3d.utility.Vector3dVector(self.explored_space.cpu()+ torch.tensor([0, 0, 0.01]))
        self.explored_space_o3d.paint_uniform_color([197/255, 237/255, 96/255])
        self.vis.update_geometry(self.explored_space_o3d)
        for i, name in enumerate(self.agents_m):
            agent = self.agents_m[name]
            self.car_model[i].points = o3d.utility.Vector3dVector(agent.car_model_mat.cpu() + torch.tensor([0, 0, 0.05]))
            if i%1 == 0:
                self.car_model[i].paint_uniform_color([245/255, 108/255, 108/255])
            elif i%3 == 1:
                self.car_model[i].paint_uniform_color([253/255, 210/255, 224/255])
            elif i%3 == 2:
                self.car_model[i].paint_uniform_color([245/255, 150/255, 125/255])
            self.vis_bound[i].points = o3d.utility.Vector3dVector(agent.detected_bound.cpu() + torch.tensor([0, 0, 0.01]))
            self.vis_bound[i].paint_uniform_color([0.6, 0.6, 0.6]) 
            self.vis.update_geometry(self.car_model[i])
            self.vis.update_geometry(self.vis_bound[i])
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.capture:
            save_path = './imgs'
            subfolders = ['all']
            # subfolders = ['all', 'agent_0', 'agent_1', 'agent_2']
            if not os.path.exists(save_path):
                print('save in: '+ os.getcwd())
                os.makedirs(save_path)
                print(f"Folder '{save_path}' created!")
                for subfolder in subfolders:
                    os.makedirs(os.path.join(save_path, subfolder))
            self.save_screenshot()
        return True
   

    def save_screenshot(self):
        """
        Saving the images for exploration process when testing
        """
        filename = f"./imgs/all/screenshot_{self.infos['local_Steps']}.png"
        img = self.vis.capture_screen_float_buffer(True)
        img = (np.asarray(img)*255).astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imwrite(filename, img)
        print(f"Saved screenshot: {filename}")   
  
    def load_map(self, map_count = 1):
        """
        load point cloud based map 
        """
        self.map_Freespace = []
        self.map_Obstacles = []
        self.map_Boundary = []
        self.map_W = []
        self.map_H = []
        for i in range(map_count):
            self.map_Freespace.append(torch.from_numpy(np.load("/data/zsh/Final_MAexp/map/"+self.config['scene']+"/"+self.config['map_list'][i]+"_freespace.npy")).float())
            self.map_Obstacles.append(torch.from_numpy(np.load("/data/zsh/Final_MAexp/map/"+self.config['scene']+"/"+self.config['map_list'][i]+"_obstacles.npy")).float())
            self.map_Boundary.append(torch.from_numpy(np.load("/data/zsh/Final_MAexp/map/"+self.config['scene']+"/"+self.config['map_list'][i]+"_boundary.npy")).float())   
            if self.config['scene'] in ['maze', 'random']:
                self.map_W.append(torch.tensor(125))
                self.map_H.append(torch.tensor(125))
            elif self.config['scene'] == 'indoor':
                map = np.load("/data/zsh/Final_MAexp/map/indoor/"+self.config['map_list'][i]+"_map.npy")
                self.map_W.append(torch.tensor(map.shape[0]/2 * self.config['map_resolution']))
                self.map_H.append(torch.tensor(map.shape[1]/2 * self.config['map_resolution']))
            elif self.config['scene'] == 'outdoor':
                self.map_W.append(torch.max(self.map_Obstacles[i][:,0]))
                self.map_H.append(torch.max(self.map_Obstacles[i][:,1]))

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
    parser.add_argument("--yaml_file", type=str, default = "./yaml/indoor.yaml", help="the yaml file for the experiment parameter")
    parser.add_argument('--is_capture', action='store_true', default=False, help="by default False, whether need to save the images.")
    """
    map config
    """
    parser.add_argument('--training_map_num', type=int, default=1, help="the number of different maps in training")
    parser.add_argument('--map_resolution', type=float, default=1.5, help="the resolution of the maps in training")
    parser.add_argument('--region', type=int, default=8, help="the number of disperse region will be divided of a map")
    parser.add_argument('--max_global_step', type=int, default=20, help="the number of global step in an exploration episode")
    parser.add_argument("--scene", type=str, default = "random", choices = ["random", "maze", "indoor", "outdoor"], help="choice the scene to explore")
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
    setup_seed(3)
    env_config = get_config()
    # register new env
    ENV_REGISTRY["maexp"] = Multiagent_exploration
    COOP_ENV_REGISTRY["maexp"] = Multiagent_exploration
    # initialize env
    env = make_env(scene_params = env_config)
    algo_class = getattr(marl.algos, env_config['algo'])
    method = algo_class(hyperparam_source="common")
    # customize model
    model = build_model(env, method, {"core_arch": "att"})
    # start learning
    if env_config['is_train'] == True:
        method.fit(env, model, stop={'episode_reward_mean': 200000, 'timesteps_total': 10000000}, local_mode = False, 
              num_workers = 4, share_policy='all', checkpoint_freq=300)
    else:
        method.fit(env, model, stop={'episode_reward_mean': 200000, 'timesteps_total': 10000000}, restore_path={'params_path': "/data/zsh/Final_MAexp/exp_results/vda2c_att_MAexp/VDA2CTrainer_maexp_MAexp_06ee7_00000_0_2024-04-19_16-15-48/params.json",  # experiment configuration
                            'model_path': "/data/zsh/Final_MAexp/exp_results/vda2c_att_MAexp/VDA2CTrainer_maexp_MAexp_06ee7_00000_0_2024-04-19_16-15-48/checkpoint_005000/checkpoint-5000"}, local_mode=True, 
                num_workers = 0, share_policy='all')
   


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