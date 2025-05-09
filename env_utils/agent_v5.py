import copy
import time
import numpy as np
import math
import env_utils.maze as maze
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL
import cv2
from einops import rearrange

class Agent_explorer:
    def __init__(self, agent_id, agent_state, env_config):
        """
        AgentExplorer 类代表一个探索环境中的智能体。
        它负责处理智能体的状态更新、动作执行、环境观测等功能。

        参数:
        - agent_id: 智能体的ID。
        - agent_state: 智能体的初始状态。
        - env_config: 环境配置参数。
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
        self.discrete_map_w = env_config['map_real_w'] / env_config['map_resolution']
        self.discrete_map_h = env_config['map_real_h'] / env_config['map_resolution']
        self.x_resolution = (self.xmax - self.xmin) / 127 # for continue scene 
        self.y_resolution = (self.ymax - self.ymin) / 127
        self.map_resolution = env_config['map_resolution'] # for discrete scene
        self.obs_size = 128
        self.transform = transforms.Resize((self.obs_size, self.obs_size))
        self.previous_neb_obs = {}
        if env_config['scene'] in ['random', 'maze', 'indoor', 'maze9', 'random2','maze_4_change', 'random3']:
            self.env_type = 'D'
            self.obs = torch.zeros((3, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        else:
            self.env_type = 'C'
            self.obs = torch.zeros((3, 128, 128), dtype = torch.float32, device=self.device)
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
        self.img= torch.zeros((6, self.obs_size, self.obs_size), dtype = torch.float32, device=self.device) # first 3 is local,latter is global
        """
        Agent State: [x, y, vel, theta]
        """
        self.agent_state = agent_state
        self.agent_state_prev = self.agent_state.clone()
        self.car_model_mat_origin = self.spawn_car_model_mat() 
        self.car_model_mat = self.trans_car_model_mat()
        
        
    def update_agent(self, env_msg):
        """
        根据环境信息更新智能体状态。
        
        参数:
        - env_msg: 环境传递的信息。
        """
        self.all_agent_model_mats = env_msg['all_agent_model_mats']
        self.is_destroy = env_msg['all_agent_collision'][self.agent_id]
        # 更新Ground Truth
        self.map_gt = self.map_obstacles
        self.other_agent_mats = torch.cat((self.all_agent_model_mats[:(self.agent_id) * self.num_carpoints], self.all_agent_model_mats[(self.agent_id+1) * self.num_carpoints:]), dim = 0)
        self.map_gt = torch.cat((self.map_gt, self.other_agent_mats), dim = 0)
        

    def step(self, action):
        # 获取当前状态
        z = self.agent_state
        delta = action[1]
        # 计算下一状态
        if z[2] > self.max_speed and action[0] > 0:
            action[0] = 0
        elif z[2] < -self.max_speed and action[0] < 0:
            action[0] = 0
        A, B, C = self.get_linear_model_matrix(z[2], z[3], delta)
        self.agent_state = torch.matmul(A, z) + torch.matmul(B, action) + C
        self.agent_state[3] = self.normalize_angle(self.agent_state[3])
        # 更新小车模型
        self.car_model_mat = self.trans_car_model_mat(self.agent_state)
        # 更新探索空间
        # self.get_observation()
        
    def spawn_car_model_mat(self):
        """
        在环境中生成车辆点云模型 在原点处
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
        将小车点云通过旋转和平移至state的状态
        points: numpy array of shape (m, n, 3)
        x: numpy array of shape (5,)
        y: numpy array of shape (5,)
        theta: numpy array of shape (5,)
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
        获取线性状态空间模型矩阵
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

    def calculate_local_map(self, image):
        height, width = image.shape[1:]
        canvas_size = max(width, height) * 2
        canvas = np.zeros((canvas_size, canvas_size, 3))
        canvas[:,:,1] = 0.5
        offset_x, offset_y = (canvas_size - width) // 2, (canvas_size - height) // 2
        pick = rearrange(image, 'c w h -> w h c').cpu().numpy()
        canvas[offset_y:offset_y + height, offset_x:offset_x + width] = pick
        ct = self.agent_state.cpu()
        agent_center = (int(ct[1]/self.map_resolution + offset_y), int(ct[0]/self.map_resolution + offset_x))
        if self.env_type == 'D':
            agent_map = canvas[agent_center[1] - 62 : agent_center[1] + 62, agent_center[0] - 62 : agent_center[0] + 62]
        else:
            agent_map = canvas[agent_center[1] - 64 : agent_center[1] + 64, agent_center[0] - 64 : agent_center[0] + 64]
        image = rearrange(torch.tensor(agent_map), 'w h c -> c w h')
        image = (self.transform(image) - self.mean.view(3, 1, 1))/self.std.view(3, 1, 1)
        return image
            
    def get_observation(self, commu):
        # 碰撞环境观测
        x, y, theta = self.get_position()
        self.car_pos = torch.tensor([x, y, 0], device=self.device)  
        distances = torch.sqrt(torch.sum((self.map_gt - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        obstacle_selected_points_mat = self.map_gt[idx].clone() # 储存当前帧检测得到的障碍    
        # 探索区域观测
        distances = torch.sqrt(torch.sum((self.map_freespace - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        self.freespace_selected_points_mat = self.map_freespace[idx].clone()  
        """
        将被遮挡的freespace -> obstacle
        """
        # 计算矢量
        fs_vec = self.freespace_selected_points_mat - self.car_pos
        ob_vec = obstacle_selected_points_mat - self.car_pos

        # Broadcast            
        fs_vec_exp = fs_vec.unsqueeze(1)  # shape: (N, 1, 3)
        ob_vec_exp = ob_vec.unsqueeze(0)  # shape: (1, M, 3)

        # 计算每个矢量的长度
        fs_vec_norm = torch.norm(fs_vec_exp, dim=2, keepdim=True)  # shape: (N, 1, 1)
        ob_vec_norm = torch.norm(ob_vec_exp, dim=2, keepdim=True)  # shape: (1, M, 1)

        cos_angle = torch.sum(fs_vec_exp * ob_vec_exp, dim=2) / torch.squeeze(fs_vec_norm * ob_vec_norm)  # shape: (N, M)

        # 角度判断      
        angle_obstacle = cos_angle > 0.99 - 1e-3 
        # 距离判断
        closer_obstacle = ob_vec_norm.reshape(1, -1) - fs_vec_norm.reshape(-1, 1) < 1e-6
        # 障碍物判断
        is_obstacle = torch.logical_and(angle_obstacle, closer_obstacle)
        # 选出被遮挡的点
        be_curtained_idx = torch.any(is_obstacle, dim=1)
        # # 将被遮挡的freespace -> obstacle
        obstacle_selected_points_mat_ = torch.cat(
            (obstacle_selected_points_mat, self.freespace_selected_points_mat[be_curtained_idx]), dim=0)

        self.freespace_selected_points_mat = self.freespace_selected_points_mat[~be_curtained_idx]
        
        self.explored_space = torch.cat((self.explored_space, self.freespace_selected_points_mat), dim=0)
        self.explored_space = torch.unique(self.explored_space, dim = 0, sorted = False)

        self.detected_bound_for_map = torch.cat((self.detected_bound, obstacle_selected_points_mat_), dim=0)
        self.detected_bound = torch.cat((self.detected_bound, obstacle_selected_points_mat), dim=0)

        self.detected_bound_for_map = torch.unique(self.detected_bound_for_map, dim = 0, sorted = False)
        self.detected_bound = torch.unique(self.detected_bound, dim = 0, sorted = False)
        # 在self.detected_bound中去掉小车的点
        if self.other_agent_mats.numel():
            diff = self.detected_bound.unsqueeze(1) - self.other_agent_mats.unsqueeze(0)
            distances = torch.norm(diff, dim=-1)
            mask = (distances.min(dim=1).values == 0)
            self.detected_bound = self.detected_bound[~mask]

        """
        将得到的点云观测值转换为栅格地图
        """
        if self.env_type == 'D':
            voxels = self.explored_space/self.map_resolution
            delete_move_obs = torch.nonzero((self.obs[0]+self.obs[1])==1)
            self.obs[1, delete_move_obs[:,0], delete_move_obs[:,1]] = 0
            self.obs[0, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            voxels = self.detected_bound/self.map_resolution
            self.obs[1, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            self.obs[1, (self.all_agent_model_mats[:, 0]/self.map_resolution).long(),
                      (self.all_agent_model_mats[:, 1]/self.map_resolution).long()] = 0
            self.obs[2] = 0
            self.obs[2, (self.car_pos[0] /self.map_resolution).long()-1:(self.car_pos[0] /self.map_resolution).long()+2,
                      (self.car_pos[1] /self.map_resolution).long()-1:(self.car_pos[1] /self.map_resolution).long()+2,] = 0.8
            neb = self.all_agent_model_mats.view(-1, 25, 3)
            neb = neb[commu[1:]].flatten(0, 1)
            self.obs[2,(neb[:,0]/self.map_resolution).long(),(neb[:,1]/self.map_resolution).long()] = 0.5 # 这里的维度还没有转化为特征图的维度
            self.img[:3] = self.transform(self.obs) # (c w h)
            # calculate local map
            self.img[3:] = self.calculate_local_map(self.obs)
            # if self.agent_id==0:
            #     a = self.img[:3]
            #     a = (rearrange(a, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
            #     cv2.imwrite('/remote-home/ums_zhushaohao/new/2024/MAexp/test.png', a)
            #     a = self.img[3:]
            #     a = (rearrange(a, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
            #     cv2.imwrite('/remote-home/ums_zhushaohao/new/2024/MAexp/test2.png', a)
        elif self.env_type == 'C':
            self.obs = torch.zeros((3, 128, 128), dtype = torch.float32, device=self.device)
            grid_map = torch.zeros((128, 128), dtype=torch.int, device=self.device)
            voxels = self.explored_space
            voxels = torch.cat((voxels, self.other_agent_mats), dim=0)
            x_indices = ((voxels[:, 0] - self.xmin) / self.x_resolution).floor().long()
            y_indices = ((voxels[:, 1] - self.ymin) / self.y_resolution).floor().long()
            x_indices = torch.clamp(x_indices, min=0, max=127)
            y_indices = torch.clamp(y_indices, min=0, max=127)
            grid_map_1d_view = grid_map.view(-1)
            grid_map_1d_view.index_add_(0, x_indices * 128 + y_indices, torch.ones(x_indices.size(0), dtype=torch.int, device=self.device))
            self.obs[0] = (grid_map >= 1).int()

            voxels = self.detected_bound_for_map
            x_indices = ((voxels[:, 0] - self.xmin) / self.x_resolution).floor().long()
            y_indices = ((voxels[:, 1] - self.ymin) / self.y_resolution).floor().long()
            x_indices = torch.clamp(x_indices, min=0, max=127)
            y_indices = torch.clamp(y_indices, min=0, max=127)
            grid_map = torch.zeros((128, 128), dtype=torch.int, device=self.device)
            grid_map_1d_view = grid_map.view(-1)
            grid_map_1d_view.index_add_(0, x_indices * 128 + y_indices, torch.ones(x_indices.size(0), dtype=torch.int, device=self.device))
            self.obs[1] = (grid_map >= 1).int()
            
            self.obs[2,...] = 0
            index = torch.floor((self.car_pos /self.x_resolution)).long()
            index = torch.clamp(index, min=0, max=127)
            self.obs[2, index[0],index[1]] = 0.8
            neb = self.all_agent_model_mats.view(-1, 25, 3)
            neb = neb[commu[1:]]
            neb = torch.floor(neb / self.x_resolution).long()
            neb = torch.clamp(neb, min=0, max=127)
            self.obs[2,neb[:,0],neb[:,1]] = 0.5
            self.img[:3] = self.obs # (c w h)
            # calculate local map
            self.img[3:] = self.calculate_local_map(self.obs)
            # a = self.img[:3]
            # a = (rearrange(a, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
            # cv2.imwrite('/remote-home/share/zsh/2024/test.png', a)
            
            # a = self.img[3:]
            # a = (rearrange(a, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
            # cv2.imwrite('/remote-home/share/zsh/2024/test1.png', a)  
        return self.img


    def global_to_local(self, global_point, car_pos):
        '''
        Calculate the local position of a point in the global map
        '''
        map_w = 125
        map_h = 125
        # 首先，将全局点相对于汽车位置进行平移
        dx = global_point[0] - car_pos[0]
        dy = global_point[1] - car_pos[1]

        # 接着，执行旋转。注意，这里的角度是正值，因为我们要逆向执行旋转
        local_x = dx * torch.cos(car_pos[-1]) + dy * torch.sin(car_pos[-1])
        local_y = dy * torch.cos(car_pos[-1]) - dx * torch.sin(car_pos[-1])

        # 将转换后的局部坐标调整到智能体坐标系的中心
        center = torch.tensor([map_w / 2, map_h / 2])
        local_x += center[0]
        local_y += center[1]

        return torch.Tensor((local_x, local_y))
    

    def Rear_Axle2Center(self, x, y, theta, is_tensor = True): 
        """
        通过车后轴中心求车中心点
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