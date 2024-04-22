import copy
import time
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL
import cv2
from einops import rearrange
import yaml
import argparse

class Turtlebot_explorer:
    def __init__(self, agent_id, agent_state, env_config):
        """
        Turtlebot_Explorer class represents a Turtlebot in an exploration environment. 
        It is responsible for handling the agent's state updates, action execution, 
        and environmental observations.

        Parameters:
        - agent_id: The ID of the agent.
        - agent_state: The initial state of the agent.
        - env_config: Environment configuration parameters.
        """

        self.device = env_config['device']
        self.DT = env_config['DT']
        self.map_obstacles = env_config['map_boundary']
        self.map_boundary = env_config['map_boundary']
        self.map_freespace = env_config['map_freespace']
        self.xmin = torch.min(self.map_obstacles[:,0])
        self.xmax = torch.max(self.map_obstacles[:,0])
        self.ymin = torch.min(self.map_obstacles[:,1])
        self.ymax = torch.max(self.map_obstacles[:,1])
        self.explored_space = torch.zeros([1,3]).to(self.device)
        self.detected_bound = torch.zeros([1,3]).to(self.device)
        self.mean = torch.Tensor([0.0474, 0.171, 0.0007])
        self.std = torch.Tensor([0.4430, 0.3323, 0.7])
        self.detected_bound_for_map = torch.zeros([1,3]).to(self.device)
        self.all_agent_position = None
        self.all_agent_model_mats = None
        self.discrete_map_w = env_config['map_real_w'] / env_config['map_resolution'] + 1e-4
        self.discrete_map_h = env_config['map_real_h'] / env_config['map_resolution'] + 1e-4
        self.x_resolution = (self.xmax - self.xmin) / 125 # for continue scene 
        self.y_resolution = (self.ymax - self.ymin) / 125
        self.map_resolution = env_config['map_resolution'] # for discrete scene
        self.obs_size = 128
        self.transform = transforms.Resize((self.obs_size, self.obs_size))
        if env_config['scene'] in ['random', 'maze', 'indoor', 'maze9', 'maze_0.1']:
            self.env_type = 'D'
        else:
            self.env_type = 'C'
        """
        Agent Configuration
        """
        self.max_speed = env_config['max_speed']
        self.agent_r = 0.35 #m
        self.agent_resolution = env_config['agent_resolution']
        self.is_lidar = env_config['is_lidar']
        self.lidar_range = env_config['lidar_range'] #m
        self.num_carpoints = self.agent_resolution**2
        self.agent_id = agent_id
        self.is_destroy = False
        self.obs = torch.zeros((3, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        self.img= torch.zeros((6, self.obs_size, self.obs_size), dtype = torch.float32, device=self.device) # first 3 is local,latter is global
        """
        Agent State: [x, y, theta]
        """
        self.agent_state = agent_state
        self.agent_state_prev = self.agent_state.clone()
        self.car_model_mat_origin = self.spawn_car_model_mat() 
        self.car_model_mat = self.trans_car_model_mat()
        
        
    def update_agent(self, env_msg):
        """
        Update agents' information
        """
        self.all_agent_model_mats = env_msg['all_agent_model_mats']
        self.is_destroy = env_msg['all_agent_collision'][self.agent_id]
        # update Ground Truth
        self.map_gt = self.map_obstacles.clone()
        self.other_agent_mats = torch.cat((self.all_agent_model_mats[:(self.agent_id) * self.num_carpoints], self.all_agent_model_mats[(self.agent_id+1) * self.num_carpoints:]), dim = 0)
        self.map_gt = torch.cat((self.map_gt, self.other_agent_mats), dim = 0)
        

    def step(self, action):
        z = self.agent_state # [x, y, theta]
        A, B, C = self.get_linear_model_matrix(z, action)
        self.agent_state = torch.matmul(A, z) + torch.matmul(B, action) + C
        self.agent_state[2] = self.normalize_angle(self.agent_state[2])
        self.car_model_mat = self.trans_car_model_mat(self.agent_state)

        
    def spawn_car_model_mat(self):
        """
        Generate a point cloud model of vehicles within the environment, located at the origin.
        """
        Theta = torch.linspace(0, 2 * torch.pi, self.num_carpoints, device=self.device)
        R = torch.Tensor([self.agent_r]).to(self.device)
        R, Theta = torch.meshgrid(R, Theta)
        X = R * torch.cos(Theta)
        Y = R * torch.sin(Theta)
        car_model_mat = torch.stack((X.reshape(-1), Y.reshape(-1), torch.zeros(self.num_carpoints, device=self.device))).T
        X = torch.linspace(0, self.agent_r, self.agent_resolution, device=self.device)
        Y = torch.zeros_like(X, device = self.device)
        angle_mat = torch.stack((X.reshape(-1), Y.reshape(-1), torch.zeros(self.agent_resolution, device=self.device))).T
        car_model_mat = torch.cat((car_model_mat, angle_mat), dim = 0)
        self.num_carpoints += self.agent_resolution
        return car_model_mat
        

    def get_position(self, agent_state=None):
        
        if agent_state is None:
            agent_state = self.agent_state
        
        x = agent_state[0] 
        y = agent_state[1] 
        theta = agent_state[2]
        
        return x, y, theta

    def trans_car_model_mat(self, agent_state=None):
        """
        Translate and rotate the car's point cloud to match the state.
        """
        if agent_state is None:
            agent_state = self.agent_state
        
        x, y, theta = self.get_position()
        
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ], device=self.device)
        p = torch.tensor([x, y, 0], device=self.device)
        car_model_mats = torch.matmul(R, self.car_model_mat_origin.T).T + p

        return car_model_mats
    
    def get_linear_model_matrix(self, state, action, is_tensor = True):
        """
        Retrieve the linear state space model matrix.
        """
        v = action[0]
        theta = state[2]
        if is_tensor:
            A = torch.zeros((3, 3), device=self.device)
            A[0, 0] = 1.0
            A[1, 1] = 1.0
            A[2, 2] = 1.0
            A[0, 2] = - self.DT * v * torch.sin(theta)
            A[1, 2] = self.DT * v * torch.cos(theta)

            B = torch.zeros((3, 2), device=self.device)
            B[0, 0] = torch.cos(theta) * self.DT
            B[1, 0] = torch.sin(theta) * self.DT
            B[2, 1] = self.DT
            
            C = torch.zeros((3), device=self.device)
            C[0] = self.DT * v * torch.sin(theta) * theta
            C[1] = - self.DT * v * torch.cos(theta) * theta
            
            if self.is_destroy:
                A = torch.eye(3, device=self.device)
                B = torch.zeros((3, 2), device=self.device)
                C = torch.zeros((3), device=self.device)
        # else:
        #     A = np.zeros((4, 4))
        #     A[0, 0] = 1.0
        #     A[1, 1] = 1.0
        #     A[2, 2] = 1.0
        #     A[3, 3] = 1.0
        #     A[0, 2] = self.DT * np.cos(phi)
        #     A[0, 3] = - self.DT * v * np.sin(phi)
        #     A[1, 2] = self.DT * np.sin(phi)
        #     A[1, 3] = self.DT * v * np.cos(phi)
        #     A[3, 2] = self.DT * np.tan(delta) / self.car_wide

        #     B = np.zeros((4, 2))
        #     B[2, 0] = self.DT
        #     B[3, 1] = self.DT * v / (self.car_wide * np.cos(delta) ** 2)

        #     C = np.zeros(4)
        #     C[0] = self.DT * v * np.sin(phi) * phi
        #     C[1] = - self.DT * v * np.cos(phi) * phi
        #     C[3] = - self.DT * v * delta / (self.car_wide * np.cos(delta) ** 2)

        #     if self.is_destroy:
        #         A = np.eye(4)
        #         B = np.zeros((4, 2))
        #         C = np.zeros(4)

        return A, B, C

    def calculate_local_map(self, image):
        height, width = image.shape[1:]
        canvas_size = max(width, height) * 3
        canvas = np.zeros((canvas_size, canvas_size, 3))
        canvas[:,:,1] = 0.5
        offset_x, offset_y = (canvas_size - width) // 2, (canvas_size - height) // 2
        pick = rearrange(image, 'c w h -> w h c').cpu().numpy()
        canvas[offset_y:offset_y + height, offset_x:offset_x + width] = pick
        ct = self.agent_state.cpu()
        agent_center = (int(ct[1]/self.map_resolution + offset_y), int(ct[0]/self.map_resolution + offset_x))
        agent_map = canvas[agent_center[1] - 62 : agent_center[1] + 62, agent_center[0] - 62 : agent_center[0] + 62]
        image = rearrange(torch.tensor(agent_map), 'w h c -> c w h')
        image = (self.transform(image) - self.mean.view(3, 1, 1))/self.std.view(3, 1, 1)
        return image
            
    def get_observation(self, commu):
        x, y, theta = self.get_position()
        self.car_pos = torch.tensor([x, y, 0], device=self.device)  
        distances = torch.sqrt(torch.sum((self.map_gt - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        obstacle_selected_points_mat = self.map_gt[idx].clone()  

        distances = torch.sqrt(torch.sum((self.map_freespace - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        self.freespace_selected_points_mat = self.map_freespace[idx].clone()  
        """
        Unseen freespace -> obstacle
        """
        fs_vec = self.freespace_selected_points_mat - self.car_pos
        ob_vec = obstacle_selected_points_mat - self.car_pos        
        fs_vec_exp = fs_vec.unsqueeze(1)  # shape: (N, 1, 3)
        ob_vec_exp = ob_vec.unsqueeze(0)  # shape: (1, M, 3)
        fs_vec_norm = torch.norm(fs_vec_exp, dim=2, keepdim=True)  # shape: (N, 1, 1)
        ob_vec_norm = torch.norm(ob_vec_exp, dim=2, keepdim=True)  # shape: (1, M, 1)
        cos_angle = torch.sum(fs_vec_exp * ob_vec_exp, dim=2) / torch.squeeze(fs_vec_norm * ob_vec_norm)  # shape: (N, M)
        angle_obstacle = cos_angle > 0.998
        closer_obstacle = ob_vec_norm.reshape(1, -1) - fs_vec_norm.reshape(-1, 1) < 1e-6
        is_obstacle = torch.logical_and(angle_obstacle, closer_obstacle)
        be_curtained_idx = torch.any(is_obstacle, dim=1)
        obstacle_selected_points_mat_ = torch.cat(
            (obstacle_selected_points_mat, self.freespace_selected_points_mat[be_curtained_idx]), dim=0)

        self.freespace_selected_points_mat = self.freespace_selected_points_mat[~be_curtained_idx]
        
        self.explored_space = torch.cat((self.explored_space, self.freespace_selected_points_mat), dim=0)
        self.explored_space = torch.unique(self.explored_space, dim = 0, sorted = False)

        self.detected_bound_for_map = torch.cat((self.detected_bound, obstacle_selected_points_mat_), dim=0)
        self.detected_bound = torch.cat((self.detected_bound, obstacle_selected_points_mat), dim=0)

        self.detected_bound_for_map = torch.unique(self.detected_bound_for_map, dim = 0, sorted = False)
        self.detected_bound = torch.unique(self.detected_bound, dim = 0, sorted = False)

        if self.other_agent_mats.numel():
            diff = self.detected_bound.unsqueeze(1) - self.other_agent_mats.unsqueeze(0)
            distances = torch.norm(diff, dim=-1)
            mask = (distances.min(dim=1).values == 0)
            self.detected_bound = self.detected_bound[~mask]

        """
        point cloud -> grid map
        """
        if self.env_type == 'D':
            voxels = self.explored_space/self.map_resolution
            delete_move_obs = torch.nonzero((self.obs[0]+self.obs[1])==1)
            self.obs[1, delete_move_obs[:,0], delete_move_obs[:,1]] = 0
            self.obs[0, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            voxels = self.detected_bound/self.map_resolution
            self.obs[1, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            self.obs[1, (self.all_agent_model_mats[:, 0]/self.map_resolution).long(), (self.all_agent_model_mats[:, 1]/self.map_resolution).long()] = 0
            self.obs[2] = 0
            self.obs[2, (self.car_pos[0] /self.map_resolution).long()-1:(self.car_pos[0] /self.map_resolution).long()+2,
                      (self.car_pos[1] /self.map_resolution).long()-1:(self.car_pos[1] /self.map_resolution).long()+2,] = 0.8
            mask = commu.repeat_interleave(self.num_carpoints)
            neb = self.other_agent_mats[mask]
            self.obs[2,neb[:,0].long(),neb[:,1].long()] = 0.5
            self.img[:3] = self.transform(self.obs) # (c w h)
            # calculate local map
            self.img[3:] = self.calculate_local_map(self.obs)
            # if self.agent_id==0:
            #     a = self.img[:3]
            #     a = (rearrange(a, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
            #     a[102,75,:] = 255
            #     cv2.imwrite('/remote-home/ums_zhushaohao/2023/Multi-agent-Exploration/test.png', a)
            #     a = self.img[3:]
            #     # b = self.global_to_local(torch.tensor([102,75]),ct).int()
            #     a = (rearrange(a, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
            #     # a[b[0], b[1], :] = 255
        elif self.env_type == 'C':
            grid_map = torch.zeros((125, 125), dtype=torch.int, device=self.device)
            voxels = self.explored_space
            x_indices = ((voxels[:, 0] - self.xmin) / self.x_resolution).floor().long()
            y_indices = ((voxels[:, 1] - self.ymin) / self.y_resolution).floor().long()
            grid_map_1d_view = grid_map.view(-1)
            grid_map_1d_view.index_add_(0, x_indices * 125 + y_indices, torch.ones(x_indices.size(0), dtype=torch.int, device=self.device))
            self.img[0] = (grid_map >= 1).int()
            voxels = self.detected_bound_for_map
            x_indices = ((voxels[:, 0] - self.xmin) / self.x_resolution).floor().long()
            y_indices = ((voxels[:, 1] - self.ymin) / self.y_resolution).floor().long()
            grid_map = torch.zeros((125, 125), dtype=torch.int, device=self.device)
            grid_map_1d_view = grid_map.view(-1)
            grid_map_1d_view.index_add_(0, x_indices * 125 + y_indices, torch.ones(x_indices.size(0), dtype=torch.int, device=self.device))
            self.img[1] = (grid_map >= 1).int()
            x = (self.car_model_mat[:, 0]/self.x_resolution).clamp(min=0, max=124).long()
            y = (self.car_model_mat[:, 1]/self.y_resolution).clamp(min=0, max=124).long()
            mask = (self.img[0] == 1) & (self.img[1] == 1)
            self.img[1][mask] = 0
            self.img[3] = self.img[3] * 0.99 
            self.img[2:, x, y] = 1   
        return self.img


    def global_to_local(self, global_point, car_pos):
        '''
        Calculate the local position of a point in the global map
        '''
        map_w = 125
        map_h = 125
        dx = global_point[0] - car_pos[0]
        dy = global_point[1] - car_pos[1]
        local_x = dx * torch.cos(car_pos[-1]) + dy * torch.sin(car_pos[-1])
        local_y = dy * torch.cos(car_pos[-1]) - dx * torch.sin(car_pos[-1])
        center = torch.tensor([map_w / 2, map_h / 2])
        local_x += center[0]
        local_y += center[1]

        return torch.Tensor((local_x, local_y))
    

    def Rear_Axle2Center(self, x, y, theta, is_tensor = True): 
        """
        Determine the center of the car using the rear axle center.
        """
        if is_tensor:
            cx = x + 0.5*self.agent_length*torch.cos(theta)
            cy = y + 0.5*self.agent_length*torch.sin(theta)
        else:
            cx = x + 0.5*self.agent_length*np.cos(theta)
            cy = y + 0.5*self.agent_length*np.sin(theta)
        return cx, cy
    
    def normalize_angle(self, angle):
        normalized_angle = angle % (2 * math.pi)
        if normalized_angle >= math.pi:
            normalized_angle -= 2 * math.pi
        elif normalized_angle < -math.pi:
            normalized_angle += 2 * math.pi
        return normalized_angle