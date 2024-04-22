import copy
import time
import numpy as np
import math
import env_utils.maze as maze
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL
from einops import rearrange
import cv2


class Agent_explorer:
    def __init__(self, agent_id, agent_state, env_config):
        """
        The Agent_Explorer class represents an agent in an exploration environment. 
        It is responsible for handling the agent's state updates, action execution, 
        and environmental observations.

        Parameters:
        - agent_id: The ID of the agent.
        - agent_state: The initial state of the agent.
        - env_config: Environment configuration parameters.
        """
        self.device = env_config['device']
        self.DT = env_config['DT']
        self.map_obstacles = env_config['map_obstacles']
        self.map_boundary = env_config['map_boundary']
        self.map_freespace = env_config['map_freespace']
        self.xmin = torch.min(self.map_obstacles[:,0])
        self.xmax = torch.max(self.map_obstacles[:,0])
        self.ymin = torch.min(self.map_obstacles[:,1])
        self.ymax = torch.max(self.map_obstacles[:,1])
        self.explored_space = torch.zeros([1,3]).to(self.device)
        self.detected_bound = torch.zeros([1,3]).to(self.device)
        self.detected_bound_for_map = torch.zeros([1,3]).to(self.device)
        self.all_agent_position = None
        self.all_agent_model_mats = None
        self.discrete_map_w = env_config['map_real_w'] / env_config['map_resolution']
        self.discrete_map_h = env_config['map_real_h'] / env_config['map_resolution']
        self.x_resolution = (self.xmax - self.xmin) / 125 # for continue scene 
        self.y_resolution = (self.ymax - self.ymin) / 125
        self.map_resolution = env_config['map_resolution'] # for discrete scene
        self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((125, 125)),
                        transforms.ToTensor()
                    ])
        if env_config['scene'] in ['random', 'maze', 'indoor']:
            self.env_type = 'D'
        else:
            self.env_type = 'C'
        """
        Agent Configuration
        """
        self.max_speed = env_config['max_speed']
        self.agent_length = env_config['agent_length']
        self.car_wide = env_config['agent_wide'] 
        self.agent_resolution = env_config['agent_resolution']
        self.is_lidar = env_config['is_lidar']
        self.lidar_range = env_config['lidar_range']
        self.num_carpoints = self.agent_resolution**2
        self.agent_id = agent_id
        self.is_destroy = False
        self.obs = torch.zeros((4, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        self.img= torch.zeros((4, 125, 125), dtype = torch.int32, device=self.device)
        """
        Agent State: [x, y, vel, theta]
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
        # 更新Ground Truth
        self.map_gt = self.map_obstacles.clone()
        self.other_agent_mats = torch.cat((self.all_agent_model_mats[:(self.agent_id) * self.num_carpoints], self.all_agent_model_mats[(self.agent_id+1) * self.num_carpoints:]), dim = 0)
        self.map_gt = torch.cat((self.map_gt, self.other_agent_mats), dim = 0)
        

    def step(self, action):
        z = self.agent_state
        delta = action[1]
        if z[2] > self.max_speed and action[0] > 0:
            action[0] = 0
        elif z[2] < -self.max_speed and action[0] < 0:
            action[0] = 0
        A, B, C = self.get_linear_model_matrix(z[2], z[3], delta)
        self.agent_state = torch.matmul(A, z) + torch.matmul(B, action) + C
        self.agent_state[3] = self.normalize_angle(self.agent_state[3])
        self.car_model_mat = self.trans_car_model_mat(self.agent_state)
        
    def spawn_car_model_mat(self):
        """
        Generate a point cloud model of vehicles within the environment, located at the origin.
        """
        # 生成三角形内部的点云
        X, Y = torch.meshgrid(
            torch.linspace(-self.agent_length / 2, self.agent_length / 2, self.agent_resolution, device=self.device), 
            torch.linspace(-self.car_wide / 2, self.car_wide / 2, self.agent_resolution, device=self.device)
        )
        
        # 生成小车点云矩阵
        car_model_mat = torch.stack((X.reshape(-1), Y.reshape(-1), torch.zeros(self.agent_resolution ** 2, device=self.device))).T
        return car_model_mat

    def get_position(self, agent_state=None):
        
        if agent_state is None:
            agent_state = self.agent_state
        
        theta = agent_state[3]
        
        x = agent_state[0] + 0.5*self.agent_length*torch.cos(theta)
        y = agent_state[1] + 0.5*self.agent_length*torch.sin(theta)
        theta = agent_state[3]
        
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
    
    def get_linear_model_matrix(self, v, phi, delta, is_tensor = True):
        """
        Retrieve the linear state space model matrix.
        """
        if is_tensor:
            A = torch.zeros((4, 4), device=self.device)
            A[0, 0] = 1.0
            A[1, 1] = 1.0
            A[2, 2] = 1.0
            A[3, 3] = 1.0
            A[0, 2] = self.DT * torch.cos(phi)
            A[0, 3] = - self.DT * v * torch.sin(phi)
            A[1, 2] = self.DT * torch.sin(phi)
            A[1, 3] = self.DT * v * torch.cos(phi)
            A[3, 2] = self.DT * torch.tan(delta) / self.car_wide

            B = torch.zeros((4, 2), device=self.device)
            B[2, 0] = self.DT
            B[3, 1] = self.DT * v / (self.car_wide * torch.cos(delta) ** 2)

            C = torch.zeros((4), device=self.device)
            C[0] = self.DT * v * torch.sin(phi) * phi
            C[1] = - self.DT * v * torch.cos(phi) * phi
            C[3] = - self.DT * v * delta / (self.car_wide * torch.cos(delta) ** 2)
            
            if self.is_destroy:
                A = torch.eye(4, device=self.device)
                B = torch.zeros((4, 2), device=self.device)
                C = torch.zeros((4), device=self.device)
        else:
            A = np.zeros((4, 4))
            A[0, 0] = 1.0
            A[1, 1] = 1.0
            A[2, 2] = 1.0
            A[3, 3] = 1.0
            A[0, 2] = self.DT * np.cos(phi)
            A[0, 3] = - self.DT * v * np.sin(phi)
            A[1, 2] = self.DT * np.sin(phi)
            A[1, 3] = self.DT * v * np.cos(phi)
            A[3, 2] = self.DT * np.tan(delta) / self.car_wide

            B = np.zeros((4, 2))
            B[2, 0] = self.DT
            B[3, 1] = self.DT * v / (self.car_wide * np.cos(delta) ** 2)

            C = np.zeros(4)
            C[0] = self.DT * v * np.sin(phi) * phi
            C[1] = - self.DT * v * np.cos(phi) * phi
            C[3] = - self.DT * v * delta / (self.car_wide * np.cos(delta) ** 2)

            if self.is_destroy:
                A = np.eye(4)
                B = np.zeros((4, 2))
                C = np.zeros(4)

        return A, B, C


            
    def get_observation(self):
        x, y, theta = self.get_position()
        self.car_pos = torch.tensor([x, y, 0], device=self.device)  
        distances = torch.sqrt(torch.sum((self.map_gt - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        obstacle_selected_points_mat = self.map_gt[idx]  
        distances = torch.sqrt(torch.sum((self.map_freespace - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        self.freespace_selected_points_mat = self.map_freespace[idx]  
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
        angle_obstacle = cos_angle > 0.99 - 1e-3 
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
            delete_move_obs = torch.nonzero((self.obs[0]+self.obs[1])==2)
            self.obs[1, delete_move_obs[:,0], delete_move_obs[:,1]] = 0
            self.obs[0, voxels[:, 0].long(), voxels[:, 1].long()] = 1
            voxels = self.detected_bound_for_map/self.map_resolution
            self.obs[1, voxels[:, 0].long(), voxels[:, 1].long()] = 1
            self.obs[1, (self.car_model_mat[:, 0]/self.map_resolution).long(), (self.car_model_mat[:, 1]/self.map_resolution).long()] = 0
            self.obs[3] = self.obs[3] * 0.99 # 更新轨迹        
            self.obs[2:, (self.car_model_mat[:, 0]/self.map_resolution).long(), (self.car_model_mat[:, 1]/self.map_resolution).long()] = 1 # 储存位置
            self.img = torch.zeros((4, 125, 125), device=self.device, dtype=torch.int32)
            for i in range(len(self.obs)):
                img = self.obs[i]
                img = self.transform(img)
                self.img[i] = img[0].to(device=self.device, dtype=torch.int32)
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
            self.img[3] = self.img[3] * 0.99 # 更新轨迹
            self.img[2:, x, y] = 1 # 储存位置
        # # 可视化
        # if self.agent_id == 0:
        #     plt.imshow(self.img[0].cpu(), cmap='gray')
        #     plt.imshow(self.img[1].cpu(), cmap='gray')
        #     plt.imshow(self.img[2].cpu(), cmap='gray')
        #     plt.imshow(self.img[3].cpu(), cmap='gray')
        #     plt.show()         
        # voxels = self.map_obstacles/self.map_resolution
        # self.obs[0, voxels[:, 0].long(), voxels[:, 1].long()] = 1
        # self.obs[:, 82,  100] = 1
        # self.obs[:, 69,  17] = 1
        # self.obs[:, 21,  44] = 1
        # a = (rearrange(self.obs[:3], 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
        return self.img

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