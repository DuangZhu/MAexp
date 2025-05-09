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
        self.map_obstacles = env_config['map_boundary']
        self.map_boundary = env_config['map_boundary']
        self.map_freespace = env_config['map_freespace']
        self.xmin = torch.min(self.map_obstacles[:,0])
        self.xmax = torch.max(self.map_obstacles[:,0])
        self.ymin = torch.min(self.map_obstacles[:,1])
        self.ymax = torch.max(self.map_obstacles[:,1])
        self.explored_space = None
        self.detected_bound = None
        self.detected_bound_for_map = None
        self.all_agent_position = None
        self.all_agent_model_mats = None
        self.discrete_map_w = env_config['map_real_w'] / env_config['map_resolution']
        self.discrete_map_h = env_config['map_real_h'] / env_config['map_resolution']
        self.x_resolution = (self.xmax - self.xmin) / 125 # for continue scene 
        self.y_resolution = (self.ymax - self.ymin) / 125
        self.map_resolution = env_config['map_resolution'] # for discrete scene
        self.obs_size = 128
        self.transform = transforms.Resize((self.obs_size, self.obs_size))
        self.transform_maexp = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((128, 128)),
                        transforms.ToTensor()
                    ])
        self.make_data = True
        if env_config['scene'] in ['random', 'maze', 'indoor', 'maze9', 'random2','maze_4_change','random3']:
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
        self.obs = torch.zeros((2, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        self.tra = torch.zeros((2, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        self.goal_map = torch.zeros((2, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        """
        Agent State: [x, y, vel, theta]
        """
        self.agent_state = agent_state
        self.agent_state_prev = self.agent_state.clone()
        self.car_model_mat_origin = self.spawn_car_model_mat() 
        self.car_model_mat = self.trans_car_model_mat()
        self.local_step = 0
        self.img = torch.zeros((3, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
        self.make_data_map = None    
        self.out = None
            
    def update_agent(self, env_msg):
        """
        Update agents' information
        """
        self.all_agent_model_mats = env_msg['all_agent_model_mats']
        self.is_destroy = env_msg['all_agent_collision'][self.agent_id]
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

        X, Y = torch.meshgrid(
            torch.linspace(-self.agent_length / 2, self.agent_length / 2, self.agent_resolution, device=self.device), 
            torch.linspace(-self.car_wide / 2, self.car_wide / 2, self.agent_resolution, device=self.device)
        )
        
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
        agent_map = canvas[agent_center[1] - 62 : agent_center[1] + 62, agent_center[0] - 62 : agent_center[0] + 62]
        image = rearrange(torch.tensor(agent_map), 'w h c -> c w h')
        image = self.transform(image)
        return image

            
    def get_observation(self, commu, goal = None):
        x, y, theta = self.get_position()
        self.car_pos = torch.tensor([x, y, 0], device=self.device)  
        distances = torch.sqrt(torch.sum((self.map_gt - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        obstacle_selected_points_mat = self.map_gt[idx].clone()  
        distances = torch.sqrt(torch.sum((self.map_freespace - self.car_pos)**2, axis=1))  
        idx = distances <= self.lidar_range
        self.freespace_selected_points_mat = self.map_freespace[idx].clone()  

        # Unseen freespace -> obstacle
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
        if self.explored_space is None:
            self.explored_space = self.freespace_selected_points_mat
        else:
            self.explored_space = torch.cat((self.explored_space, self.freespace_selected_points_mat), dim=0)
        self.explored_space = torch.unique(self.explored_space, dim = 0, sorted = False)
        if self.detected_bound is None:
            self.detected_bound_for_map = obstacle_selected_points_mat_
            self.detected_bound = obstacle_selected_points_mat.clone()
        else:  
            self.detected_bound_for_map = torch.cat((self.detected_bound, obstacle_selected_points_mat_), dim=0)
            self.detected_bound = torch.cat((self.detected_bound, obstacle_selected_points_mat), dim=0)

        self.detected_bound_for_map = torch.unique(self.detected_bound_for_map, dim = 0, sorted = False)
        self.detected_bound = torch.unique(self.detected_bound, dim = 0, sorted = False)
        if self.other_agent_mats.numel():
            if not self.detected_bound.numel():
                self.detected_bound = self.map_boundary[:1,:].clone()
                self.detected_bound_for_map = self.detected_bound.clone()
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
            voxels = self.detected_bound/self.map_resolution
            self.obs[1, voxels[:, 0].long(), voxels[:, 1].long()] = 1
            self.obs[1, (self.all_agent_model_mats[:, 0]/self.map_resolution).long(), (self.all_agent_model_mats[:, 1]/self.map_resolution).long()] = 0         
            if self.make_data:
                self.tra[1] = self.tra[1] * 0.99
                self.tra[:, (self.car_pos[0] /self.map_resolution).long()-1:(self.car_pos[0] /self.map_resolution).long()+2,
                        (self.car_pos[1] /self.map_resolution).long()-1:(self.car_pos[1] /self.map_resolution).long()+2] = 1 
                # ------------------Maexp --------------------
                Maexp_input = torch.cat((self.obs, self.tra), dim = 0).clone()
                Final_Maexp_input = torch.zeros((4, 128, 128), device=self.device, dtype=torch.float32)
                for i in range(len(Maexp_input)):
                    img = Maexp_input[i]
                    img = self.transform_maexp(img)
                    Final_Maexp_input[i] = img[0].to(device=self.device, dtype=torch.float32)
                    # true map
                MAexp_true_map = torch.zeros((2, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
                voxels = self.map_freespace/self.map_resolution
                MAexp_true_map[0, voxels[:, 0].long(), voxels[:, 1].long()] = 1
                voxels = self.map_boundary/self.map_resolution
                MAexp_true_map[1, voxels[:, 0].long(), voxels[:, 1].long()] = 1
                Final_MAexp_true_map = torch.zeros((4, 128, 128), device=self.device, dtype=torch.float32)
                for i in range(len(MAexp_true_map)):
                    img = MAexp_true_map[i]
                    img = self.transform_maexp(img)
                    Final_MAexp_true_map[i] = img[0].to(device=self.device, dtype=torch.float32)
                Final_MAexp_true_map[2:] = Final_Maexp_input[2:].clone()
                # ---------------------------MAANS---------------------------------------
                MAANS_input = Final_Maexp_input.clone()
                height, width = MAANS_input.shape[1:]
                canvas_size = max(width, height) * 2
                canvas = np.zeros((canvas_size, canvas_size, 4))
                canvas[:,:,1] = 1
                offset_x, offset_y = (canvas_size - width) // 2, (canvas_size - height) // 2
                pick = rearrange(MAANS_input, 'c w h -> w h c').cpu().numpy()
                canvas[offset_y:offset_y + height, offset_x:offset_x + width] = pick
                ct = self.agent_state.cpu()
                theta_degrees = float(np.degrees(ct[-1]))
                agent_center = (int(ct[1]/self.map_resolution + offset_y), int(ct[0]/self.map_resolution + offset_x))
                M_rotate = cv2.getRotationMatrix2D(agent_center, -theta_degrees, 1)
                rotated_img = cv2.warpAffine(canvas, M_rotate, (height*2, width*2), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0.5,0)) 
                Final_MAANS_input = rotated_img[agent_center[1] - 64 : agent_center[1] + 64, agent_center[0] - 64 : agent_center[0] + 64] # image is 125*125,so 62
                Final_MAANS_input = rearrange(torch.tensor(Final_MAANS_input), 'w h c -> c w h')
                if goal is not None and self.local_step == 0:
                    goal_ = (goal[self.agent_id]/self.map_resolution).to(torch.int)
                    self.goal_map[0] = self.goal_map[0] + self.goal_map[1] # all goals
                    self.goal_map[1] = 0
                    self.goal_map[1, goal_[0]-2:goal_[0]+3, goal_[1]-2:goal_[1]+3] = 1
                    self.goal_map = torch.clamp(self.goal_map, max=1)    
                Final_MAANS_input_ = torch.zeros((2, 128, 128), device=self.device, dtype=torch.float32)
                goal_input = self.goal_map.cpu().clone()
                for i in range(len(goal_input)):
                    img = goal_input[i]
                    img = self.transform_maexp(img)
                    Final_MAANS_input_[i] = img[0].to(device=self.device, dtype=torch.float32)
                Final_MAANS_input = torch.cat((Final_MAANS_input_.cpu(), Final_MAANS_input), dim = 0).clone()
                
                
                MAANS_true_map = Final_MAexp_true_map.clone()
                height, width = MAANS_true_map.shape[1:]
                canvas_size = max(width, height) * 2
                canvas = np.zeros((canvas_size, canvas_size, 4))
                canvas[:,:,1] = 1
                offset_x, offset_y = (canvas_size - width) // 2, (canvas_size - height) // 2
                pick = rearrange(MAANS_true_map, 'c w h -> w h c').cpu().numpy()
                canvas[offset_y:offset_y + height, offset_x:offset_x + width] = pick
                ct = self.agent_state.cpu()
                theta_degrees = float(np.degrees(ct[-1]))
                agent_center = (int(ct[1]/self.map_resolution + offset_y), int(ct[0]/self.map_resolution + offset_x))
                M_rotate = cv2.getRotationMatrix2D(agent_center, -theta_degrees, 1)
                rotated_img = cv2.warpAffine(canvas, M_rotate, (height*2, width*2), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0.5,0)) 
                Final_MAANS_true_map = rotated_img[agent_center[1] - 64 : agent_center[1] + 64, agent_center[0] - 64 : agent_center[0] + 64] # image is 125*125,so 62
                Final_MAANS_true_map = rearrange(torch.tensor(Final_MAANS_true_map), 'w h c -> c w h')
            voxels = self.explored_space/self.map_resolution
            delete_move_obs = torch.nonzero((self.img[0]+self.img[1])==1)
            self.img[1, delete_move_obs[:,0], delete_move_obs[:,1]] = 0
            self.img[0, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            voxels = self.detected_bound/self.map_resolution
            self.img[1, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            self.img[1, (self.all_agent_model_mats[:, 0]/self.map_resolution).long(), (self.all_agent_model_mats[:, 1]/self.map_resolution).long()] = 0
            self.img[2] = 0
            self.img[2, (self.car_pos[0] /self.map_resolution).long()-1:(self.car_pos[0] /self.map_resolution).long()+2,
                    (self.car_pos[1] /self.map_resolution).long()-1:(self.car_pos[1] /self.map_resolution).long()+2,] = 0.8
            mask = commu.repeat_interleave(self.num_carpoints)
            neb = self.other_agent_mats[mask]
            self.img[2,neb[:,0].long(),neb[:,1].long()] = 0.5
            # calculate local map
            mask_map = self.calculate_local_map(self.img)
            # true map
            self.true_map = torch.zeros((3, self.discrete_map_w.long(), self.discrete_map_h.long()), dtype = torch.float32, device=self.device)
            voxels = self.map_freespace/self.map_resolution
            self.true_map[0, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            voxels = self.map_boundary/self.map_resolution
            self.true_map[1, voxels[:, 0].long(), voxels[:, 1].long()] = 0.5
            self.true_map[2, (self.car_pos[0] /self.map_resolution).long()-1:(self.car_pos[0] /self.map_resolution).long()+2,
                    (self.car_pos[1] /self.map_resolution).long()-1:(self.car_pos[1] /self.map_resolution).long()+2,] = 0.8
            self.true_map[2,self.other_agent_mats[:,0].long(),self.other_agent_mats[:,1].long()] = 0.5
            self.true_map = self.calculate_local_map(self.true_map)
            self.make_data_map = torch.cat((mask_map, self.true_map), dim = 0)
            if self.make_data:
                self.out = {'maexp_input': Final_Maexp_input.cpu().numpy(), 'maexp_gt': Final_MAexp_true_map.cpu().numpy(),
                    'maans_input': Final_MAANS_input.cpu().numpy(), 'maans_gt': Final_MAANS_true_map.cpu().numpy(),
                    'Ours': self.make_data_map.cpu().numpy()}
                
            else:
                self.out = {'maexp_input': [], 'maexp_gt': [],
                    'maans_input': [], 'maans_gt': [],
                    'Ours': []}
            return self.out
        
        elif self.env_type == 'C':
            assert False, "This part of the code has not been implemented yet."
        return self.obs

    def Rear_Axle2Center(self, x, y, theta, is_tensor = True): 
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