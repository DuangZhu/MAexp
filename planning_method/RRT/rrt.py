"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
author: AtsushiSakai(@Atsushi_twi)
"""

import math
import random
from sys import float_repr_style

import matplotlib.pyplot as plt


import numpy as np
import copy
import torch
from torch.distributions import Categorical

show_animation = False

def rad_fix(x):
    while x>=math.pi*2:
        x -= math.pi *2
    while x<0:
        x += math.pi*2
    return x

def in_rad_section(x, l, r):
    x = rad_fix(x)
    l = rad_fix(l)
    r = rad_fix(r)
    return abs(rad_fix(r-l) - (rad_fix(x-l) + rad_fix(r-x))) < 1e-3

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.cost = 0.0
            self.start_angle = 0.0

    def __init__(self,
                 start,
                 goals,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:list of goals [[x,y], ...]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        self.start = self.Node(start[0], start[1])
        self.end = [self.Node(x, y) for x,y in goals]
        self.num_goals = len(goals)
        self.min_rand_x = rand_area[0][0]
        self.max_rand_x = rand_area[0][1]
        self.min_rand_y = rand_area[1][0]
        self.max_rand_y = rand_area[1][1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.add_boundary()

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            if i % 100 == 0:
                print("Iter : %d, Nodes : %d"%(i, len(self.node_list)))
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                # self.node_list.append(new_node)
                self.push_node(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.get_nearest_goal(self.node_list[-1].x, self.node_list[-1].y),
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path
    
    def get_explored_map(self, goal, pad = 3):
        H, W = self.obs.shape
        ret = np.zeros((2, H, W), dtype=np.float32)
        ret[1, goal[0]-pad:goal[0]+pad+1, goal[1]-pad:goal[1]+pad+1] = 1.0
        for node in self.node_list:
            ret[0, int(node.x)-pad:int(node.x)+pad+1, int(node.y)-pad:int(node.y)+pad+1] = 1.0
        return ret
    
    def get_explorable_map(self, goal, pad = 3):
        H, W = self.obs.shape
        ret = np.zeros((2, H, W), dtype=np.float32)
        # ret[1, goal[0]-pad:goal[0]+pad+1, goal[1]-pad:goal[1]+pad+1] = 1.0
        '''for dir in self.direction:
            dx = np.cos(dir)
            dy = np.sin(dir)
            x, y = self.start.x, self.start.y
            for _ in range(100):
                if int(x)>=H or int(x)<0 or int(y)>=W or int(y)<0:
                    break
                ret[1, int(x)-1:int(x)+2, int(y)-1:int(y)+2] = 1.0
                x += dx
                y += dy'''
        seg_obs = [obj for obj in self.obstacle_list if len(obj) == 2]
        for obj in [seg_obs[4], seg_obs[5]]:
            sx, sy = obj[0]
            ex, ey = obj[1]
            for i in range(100 + 1):
                x = sx + (ex-sx) / 100 * i
                y = sy + (ey-sy) / 100 * i
                x, y = int(x), int(y)
                ret[1, int(x)-1:int(x)+2, int(y)-1:int(y)+2] = 1.0

        # dist or vis?
        from queue import Queue
        que = Queue()
        que.put([int(self.start.x), int(self.start.y)])
        ret[0][int(self.start.x), int(self.start.y)] = 1
        steps = [(-1,0), (1,0), (0,1), (0, -1)]
        sx, sy = int(self.start.x), int(self.start.y)
        while not que.empty():
            x, y = que.get()
            steps = [(-1,0), (1,0), (0,1), (0, -1)]
            if abs(x-sx) == abs(y-sy) and abs(x-sx) == ret[0][x,y]-ret[0,sx,sy]:
                # diagnal
                steps += [(1,1), (1,-1), (-1,1), (-1,-1)]
            neigh = [(x+dx,y+dy) for (dx, dy) in steps]
            neigh = [(x,y) for x,y in neigh if x>=0 and x<H and y>=0 and y<W and self.obs[x,y] != 1 and ret[0][x,y] == 0 and self.unexplored[x,y] == 0]
            for u,v in neigh:
                if self.check_collision(self.Node(x+0.5,y+0.5), seg_obs, last_node = self.Node(u+0.5, v+0.5)):
                    ret[0][u, v] = ret[0][x,y]+1
                    que.put((u,v))
        #print(ret[0].max())
        lim = 700
        mask = (ret[0] == 0) + (ret[0] > lim)
        v = lim * 1.1
        ret[0] = (v - ret[0]) / v
        ret[0][mask] = 0.0
        ret[0] = ret[0] / ret[0].max()
        return ret

    def set_obs(self, obs, unexplored):
        self.obs = obs.copy()
        self.unexplored = unexplored.copy()

    def select_frontiers(self, map, num_targets = 100, get_farthest = False, sections = None):
        H, W = map.shape
        targets = []
        if sections is not None:
            targets = [[] for _ in range(len(sections))]
        iter = 0
        self.node_list = [self.start]
        tot_targets = 0
        while iter < self.max_iter and tot_targets < num_targets:
            iter += 1
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            x, y = int(new_node.x), int(new_node.y)
            x = max(0, min(x, H-1))
            y = max(0, min(y, W-1))
            if self.check_collision(new_node, self.obstacle_list):
                new_node.start_angle = new_node.parent.start_angle
                if self.check_collision(new_node, self.obstacle_list, last_node = self.start):
                    new_node.start_angle = rad_fix(math.atan2(new_node.y - self.start.y, new_node.x - self.start.x))
                if map[x,y] == 1:
                    # unexplored
                    tot_targets += 1
                    if sections is None:
                        targets.append((x,y))
                    else:
                        for i, (l, r) in enumerate(sections):
                            if in_rad_section(new_node.start_angle, l, r):
                                targets[i].append((x,y))
                else:
                    self.push_node(new_node)
        success = len(targets) > 0 if sections is None else [len(s) > 0 for s in targets]
        if sections is not None:
            for i, (l, r) in enumerate(sections):
                p = self.start
                if get_farthest:
                    for x in self.node_list:
                        if in_rad_section(x.start_angle, l, r) and x.cost > p.cost:
                            p = x
                targets[i].append((max(0, min(int(p.x), H-1)), max(0, min(int(p.y), W-1))))
        self.targets = copy.deepcopy(targets)
        self.success = copy.deepcopy(success)
        return targets, success
    
    def generate_goal_map(self, goal):
        x, y = int(goal[0]), int (goal[1])
        ret = np.zeros_like(self.obs)
        ret[x-2:x+3,y-2:y+3] = 1.0
        return ret

    def generate_targets_map(self, targets):
        ret = np.zeros_like(self.obs)
        for x,y in targets:
            ret[x-1:x+2, y-1:y+2] = 1.0
        return ret
    
    def get_distance(self, u, v):
        return math.hypot(u.x - v.x, u.y - v.y)
    
    def add_segment(self, map, u, v):
        H, W = self.obs.shape
        for i in range(10 + 1):
            x = u.x + (v.x-u.x)/10*i
            y = u.y + (v.y-u.y)/10*i
            x = max(0, min(int(x), H-1))
            y = max(0, min(int(y), W-1))
            map[x-1:x+2,y-1:y+2] = 1.0
        return map
    
    def generate_goal_trace(self, goal):
        goal = self.Node(goal[0], goal[1])
        p = self.start
        dist = self.get_distance(goal, p)
        for x in self.node_list:
            tmp = self.get_distance(goal, x)
            if tmp < dist:
                dist = tmp
                p = x
        
        ret = np.zeros_like(self.obs)
        ret = self.add_segment(ret, goal, p)
        while self.get_distance(p, self.start) > 1e-3:
            q = p.parent
            ret = self.add_segment(ret, p, q)
            p = q
        return ret

    def generate_rrt_trace(self):
        ret = np.zeros_like(self.obs)
        for x in self.node_list:
            if self.get_distance(x, self.start) > 1e-3:
                ret = self.add_segment(ret, x, x.parent)
        return ret

    def get_segment(self, start_loc, angle):
        H, W = self.unexplored.shape
        dx, dy = np.cos(angle), np.sin(angle)
        l = 0
        r = self.max_rand_x - self.min_rand_x + self.max_rand_y - self.min_rand_y
        for _ in range(100): # binary search
            mid = (l + r) / 2
            A = self.Node(start_loc[0], start_loc[1])
            B = self.Node(start_loc[0] + dx * mid, start_loc[1] + dy * mid)
            x, y = int(start_loc[0] + dx * mid), int(start_loc[1] + dy * mid)
            if self.check_collision(A, self.obstacle_list, last_node = B) and not (x>=0 and x<H and y>=0 and y<W and self.unexplored[x,y] == 1):
                l = mid
            else:
                r = mid
        r = min(r * 0.9, 50)
        return ((start_loc[0] - dx * 0.1, start_loc[1] - dy * 0.1), (start_loc[0] + dx * r, start_loc[1] + dy * r))

    def add_boundary(self):
        for x in [self.min_rand_x, self.max_rand_x]:
            self.obstacle_list.append(((x, self.min_rand_y), (x, self.max_rand_y)))
        for y in [self.min_rand_y, self.max_rand_y]:
            self.obstacle_list.append(((self.min_rand_x, y), (self.max_rand_x, y))) 

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length >= d:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
        else:
            new_node.x += extend_length * math.cos(theta)
            new_node.y += extend_length * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        # if extend_length > d:
        #    extend_length = d

        # n_expand = math.floor(extend_length / self.path_resolution)

        #for _ in range(n_expand):
        #    new_node.x += self.path_resolution * math.cos(theta)
        #    new_node.y += self.path_resolution * math.sin(theta)
        #    new_node.path_x.append(new_node.x)
        #    new_node.path_y.append(new_node.y)

        #d, _ = self.calc_distance_and_angle(new_node, to_node)
        #if d <= self.path_resolution:
        #    new_node.path_x.append(to_node.x)
        #    new_node.path_y.append(to_node.y)
        #    new_node.x = to_node.x
        #    new_node.y = to_node.y

        new_node.parent = from_node

        new_node.cost = from_node.cost + math.hypot(new_node.x - from_node.x, new_node.y - from_node.y)

        return new_node

    def generate_final_course(self, goal_ind, ignore_goal = False):
        node = self.node_list[goal_ind]
        goal = self.get_nearest_goal(node.x, node.y)
        if ignore_goal:
            path = []
        else:
            path = [[goal.x, goal.y]]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        path = [x for x in reversed(path)]
        return path

    def calc_dist_to_goal(self, x, y):
        node = self.get_nearest_goal(x, y)
        dx = x - node.x
        dy = y - node.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand_x, self.max_rand_x),
                random.uniform(self.min_rand_y, self.max_rand_y))
        else:  # goal point sampling
            idx = random.randint(0, self.num_goals - 1)
            rnd = self.Node(self.end[idx].x, self.end[idx].y)
        return rnd
    
    def push_node(self, node):
        for p in self.node_list:
            if math.hypot(node.x - p.x, node.y - p.y) < 1e-5:
                return False
        self.node_list.append(node)
        return True

    def draw_graph(self, rnd=None, fig=None, ax=None, goal = None):
        if fig is None:
            plt.clf()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if rnd is not None:
                plt.plot(rnd.x, rnd.y, "^k")
            # print("%d nodes"%len(self.node_list))
            for node in self.node_list:
                if node.parent:
                    plt.plot(node.path_x, node.path_y, "-g")

            for obj in self.obstacle_list:
                if len(obj) == 3:
                    ox, oy, size = obj
                    self.plot_circle(ox, oy, size)
                if len(obj) == 4:
                    x1, y1, x2, y2 = obj
                    self.plot_rectangle(x1, y1, x2, y2)
                if len(obj) == 2:
                    (x1, y1), (x2, y2) = obj
                    self.plot_segment(x1, y1, x2, y2)

            plt.plot(self.start.x, self.start.y, "xr")
            for goal in self.end:
                plt.plot(goal.x, goal.y, "xb")
            plt.axis("equal")
            plt.axis([-2, 15, -2, 15])
            plt.grid(True)
            plt.pause(0.01)
        else:
            for node in self.node_list:
                if node.parent:
                    ax.plot(node.path_x, node.path_y, "-b")
            for x in range(int(self.max_rand_x)):
                for y in range(int(self.max_rand_y)):
                    if self.unexplored[x,y] != 1:
                        # explored
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, color='grey'))
            for obj in self.obstacle_list:
                if len(obj) == 3:
                    ox, oy, size = obj
                    self.plot_circle(ox, oy, size, ax = ax)
                if len(obj) == 4:
                    x1, y1, x2, y2 = obj
                    self.plot_rectangle(x1, y1, x2, y2, ax = ax)
                if len(obj) == 2:
                    (x1, y1), (x2, y2) = obj
                    self.plot_segment(x1, y1, x2, y2, ax = ax)
            ax.plot(self.start.x, self.start.y, 'xr', markersize=1)
            if goal is None:
                for goal in self.end:
                    ax.plot(goal.x, goal.y, 'or', markersize=1)
            else:
                ax.plot(goal[0], goal[1], 'or', markersize=1)
            ax.axis([self.min_rand_x-1, self.max_rand_x+1, self.min_rand_y-1, self.max_rand_y+1])
    
    def get_nearest_goal(self, x, y):
        dist = [math.hypot(n.x - x, n.y - y) for n in self.end]
        return self.end[dist.index(min(dist))]
    
    @staticmethod
    def plot_rectangle(x1, y1, x2, y2, color = "green", ax= None):
        if ax is None:
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1 , y2-y1, color = color))
        else:
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color=color))

    @staticmethod
    def plot_circle(x, y, size, color="-g", ax = None):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        if ax is None:
            plt.plot(xl, yl, color)
        else:
            ax.plot(xl, yl, color)

    @staticmethod
    def plot_segment(x1, y1, x2, y2, color = "-g", ax = None):
        x = [x1 + (x2-x1) / 100 * i for i in range(101)]
        y = [y1 + (y2-y1) / 100 * i for i in range(101)]
        if ax is None:
            plt.plot(x, y, color)
        else:
            ax.plot(x, y, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList, last_node = None):

        if node is None:
            return False
        
        if last_node == None:
            if node.parent is not None:
                last_node = node.parent
            else:
                last_node = None
        def check_intersect(px, py, qx, qy, node, last_node):
            crs = 0
            # print("(%.3f, %.3f), (%.3f, %.3f)"%(px, py, qx, qy))
            dx_last, dy_last = last_node.x - px, last_node.y - py
            dx_now, dy_now = node.x - px, node.y - py
            dx, dy = qx - px, qy - py
            crs_last = dx_last * dy - dx * dy_last
            crs_now = dx_now * dy - dx * dy_now
            # print('(%.5f, %.5f) -> (%.5f, %.5f)'%(node.parent.x, node.parent.y, node.x, node.y), crs_last, crs_now)
            if crs_last > 0 and crs_now < 0:
                crs += 1
            if crs_last < 0 and crs_now > 0:
                crs += 1
            
            dx, dy = node.x - last_node.x, node.y - last_node.y
            dx_p, dy_p = px - last_node.x, py - last_node.y
            dx_q, dy_q = qx - last_node.x, qy - last_node.y
            crs_p = dx_p * dy - dx * dy_p
            crs_q = dx_q * dy - dx * dy_q
            # print('(%.5f, %.5f) -> (%.5f, %.5f)'%(node.parent.x, node.parent.y, node.x, node.y), crs_p, crs_q)
            if crs_p > 0 and crs_q < 0:
                crs += 1
            if crs_p < 0 and crs_q > 0:
                crs += 1
            if crs == 2:
                return False
            return True
        for obj in obstacleList:
            if len(obj) == 3:
                # cycle
                ox, oy, size = obj
                dx_list = [ox - x for x in node.path_x]
                dy_list = [oy - y for y in node.path_y]
                d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

                if min(d_list) <= size**2:
                    return False  # collision
            if len(obj) == 4:
                # rectangle
                x1, y1, x2, y2 = obj
                if node.x>=x1 and node.x<=x2 and node.y>=y1 and node.y<=y2:
                    # inside
                    return False
                if last_node == None:
                    continue
                corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                for i in range(4):
                    px, py = corners[i]
                    qx, qy = corners[(i+1)%4]
                    if not check_intersect(px, py, qx, qy, node, last_node):
                        return False
            if len(obj) == 2:
                if last_node == None:
                    continue
                (px, py), (qx, qy) = obj
                if not check_intersect(px, py, qx, qy, node, last_node):
                    return False
                    
        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def find_rectangle_obstacles(map):
    map = map.copy().astype(np.int32)
    map[map == 2] = 0
    H, W = map.shape
    obstacles = []
    covered = np.zeros((H, W), dtype = np.int32)
    pad = 0.01
    for x in range(H):
        for y in range(W):
            if map[x,y] == 1 and covered[x,y] == 0:
                x1 = x
                x2 = x
                while x2 < H-1 and map[x2 + 1, y] == 1:
                    x2 = x2 + 1
                y1 = y
                y2 = y
                while y2 < W-1 and map[x1 : x2+1, y2 + 1].sum() == x2-x1+1:
                    y2 = y2 + 1
                covered[x1 : x2 + 1, y1 : y2 + 1] = 1
                obstacles.append((x1-pad, y1-pad, x2 + 1 + pad, y2 + 1 + pad))    
    return obstacles

def circle_matrix(H, W, p, radius):
    mat = np.zeros((H, W), dtype = np.int32)
    for x in range(H):
        for y in range(W):
            if math.hypot(x-p[0], y-p[1])<= radius:
                mat[x,y] = 1
    return mat

class WMA_RRT:
    class Node:
        def __init__(self, x, y, parent=None):
            self.x = x
            self.y = y
            self.parent = parent
            self.childs = []
            self.id = random.randint(0, 2147483647) # a random id

            # WMA_RRT algorithm related
            self.locked = False
            self.lock_rest_time = 0
            self.keeper=-1
            self.completed = False
            self.exist = True
            self.visiters = []
            self.candidates = []
        
        def add_child(self, node):
            self.childs.append(node)
        
        def lock(self, lock_time, keeper=-1):
            self.locked = True
            self.keeper = keeper
            self.lock_rest_time = lock_time
        
        def unlock(self):
            self.locked = False
            self.lock_rest_time = 0
            self.keeper = -1

        def lock_count_down(self, down=1):
            self.lock_rest_time -= down
            if self.lock_rest_time <= 0:
                self.unlock()
        
        def position(self):
            return self.x, self.y
        
        @property
        def visited(self):
            return len(self.visiters) > 0

    def __init__(self, start, rand_area, map, unexplored, expand_dis=10, node_radius=3, max_iter=2000, lock_time=2, strict=True):
        start = [(x+0.5, y+0.5) for x,y in start]
        self.locations = start
        self.num_agents = len(start)
        self.lock_time = lock_time
        self.stage = "FindMainRoot" # switches to "Exploration" at some point
        self.main_root = self.Node(*start[0]) # use the first agent's location as main root

        [self.lx, self.rx], [self.ly, self.ry] = rand_area

        self.map = copy.deepcopy(map)
        self.unexplored = copy.deepcopy(unexplored)
        self.expand_dis = expand_dis
        self.node_radius = node_radius
        self.max_iter = max_iter
        self.step = 0
        self.nodes = []
        self.line_map = None
        self.history = np.zeros_like(self.map).astype(np.float32)
        self.strict = strict

    def init_exploration_stage(self):
        self.stage = "Exploration"
        self.agent_state = []
        self.nodes = [self.main_root] # 主节点
        '''
        agent has 3 states:
        1. at node None
        2. to nodeA nodeB
        3. try nodeA locB
        '''
        for a, loc in enumerate(self.locations):
            node = self.Node(*loc, parent=self.main_root)
            node.lock(self.lock_time)
            
            self.agent_state.append(('at', node, None))
            self.main_root.add_child(node) # 所有初始的智能体位置都会作为main_root的子节点
            self.nodes.append(node)
    
    def distance(self, nodeA, nodeB):
        return math.hypot(nodeA.x-nodeB.x, nodeA.y-nodeB.y)

    def add_node(self, loc, parent=None):
        new_node = self.Node(*loc, parent=parent)
        for node in self.nodes:
            if self.distance(node, new_node) <= self.node_radius:
                # duplicated node
                return node, False
        if (self.history[int(new_node.x)-1:int(new_node.x)+2, int(new_node.y)-1:int(new_node.y)+2] > 0).all():
            new_node.visiters.append([-1])
        if parent is not None:
            parent.add_child(new_node)
        self.nodes.append(new_node)
        return new_node, True
    
    def get_closest_node(self, nodeA):
        min_dist = 1e9
        ret = None
        for node in self.nodes:
            dist = self.distance(nodeA, node)
            if dist < min_dist:
                min_dist = dist
                ret = node
        return ret, min_dist
    
    def refresh_completed(self, node):
        # print("try refresh", node.id)
        if not node.exist:
            return True
        node.completed = (len(node.visiters)>0)
        for x in node.childs:
            # print("find son", node.id, x.id)
            node.completed = self.refresh_completed(x) and node.completed
        # print("refersh completed", node.id, node.completed, node.position(), node.parent.id if node.parent is not None else None, [x.id for x in node.childs])
        self.node_count += 1
        return node.completed

    def update_map(self, map, unexplored, locations):
        self.map = copy.deepcopy(map)
        self.unexplored = copy.deepcopy(unexplored)
        locations = [(x+0.5, y+0.5) for x, y in locations]
        if self.stage == "FindMainRoot":
            self.locations = copy.deepcopy(locations)
            if all([self.no_collision(self.Node(*loc), self.main_root) for loc in self.locations]) and all([self.distance(self.Node(*loc), self.main_root) <= self.expand_dis for loc in self.locations]):
                self.init_exploration_stage()
        else:
            for a in range(self.num_agents):
                loc0 = self.locations[a]
                loc1 = locations[a]
                if self.agent_state[a][0] == 'at':
                    node = self.agent_state[a][1]
                    # assert self.distance(self.Node(*loc1), node) <= self.expand_dis
                elif self.agent_state[a][0] == 'to':
                    nodeA, nodeB = self.agent_state[a][1:]
                    if self.distance(self.Node(*loc1), nodeB) <= self.node_radius:
                        self.agent_state[a] = ('at', nodeB, None)
                elif self.agent_state[a][0] == 'try':
                    locB = self.agent_state[a][2]
                    if self.distance(self.Node(*loc1), self.Node(*locB)) <= self.node_radius:
                        nodeB, _ = self.add_node(loc1, parent=self.agent_state[a][1])
                        self.agent_state[a] = ('at', nodeB, None)
                    # if not self.no_collision(self.Node(*loc1), self.Node(*locB)):
                    #     self.agent_state[a] = ('to', self.agent_state[a][1], self.agent_state[a][1])
                if self.agent_state[a][0] == "try":
                    print(f"agent {a}, loc {loc1}, state {self.agent_state[a][0]} {self.agent_state[a][2]}")
                elif self.agent_state[a][0] == "to":
                    print(f"agent {a},  loc {loc1}, state {self.agent_state[a][0]} {self.agent_state[a][2].position()}")
                elif self.agent_state[a][0] == "at":
                    print(f"agent {a},  loc {loc1}, state {self.agent_state[a][0]} {self.agent_state[a][1].position()}")

                for node in self.nodes:
                    if a not in node.visiters and self.distance(self.Node(*loc1), node) <= self.node_radius and self.no_collision(self.Node(*loc1), node):
                        node.visiters.append(a)
            self.locations = copy.deepcopy(locations)

            self.remove_wrong_nodes()
            print("step", self.step)
            self.node_count = 0
            self.refresh_completed(self.main_root)
            assert self.node_count == len(self.nodes), [x.id for x in self.nodes]
        
        for loc in self.locations:
            mat = circle_matrix(*self.map.shape, loc, self.node_radius)
            self.history += mat.astype(np.float32) * (1. - unexplored).astype(np.float32)

        # self.draw("gjx/wma_rrt/step{}.png".format(self.step))

    def draw(self, path):
        import imageio
        map = copy.deepcopy(self.map)
        map = (map == 1).astype(np.uint8)
        for a in range(self.num_agents):
            x, y = self.locations[a]
            x, y = int(x), int(y)
            map[x-2:x+3, y-2:y+3] = 0.
        unexplored = copy.deepcopy(self.unexplored)
        map = (map == 1).astype(np.uint8)
        unexplored[map == 1] = 0

        def add_color(a, b, cs):
            for i, c in enumerate(cs):
                a[: ,:, i] += (b * c).astype(np.uint8)
        rgb = np.zeros((*map.shape, 3)).astype(np.uint8)
        add_color(rgb, map, [0, 80, 0])
        add_color(rgb, 1 - unexplored, [40, 40, 40])
        for node in self.nodes:
            if node.parent is not None:
                x0, y0 = node.x, node.y
                x1, y1 = node.parent.x, node.parent.y
                dx, dy = x1-x0, y1-y0
                r = math.hypot(dx, dy)
                dx /= r
                dy /= r
                x, y = x0, y0
                while int(x)!=int(x1) or int(y)!=int(y1):
                    x += dx
                    y += dy
                    ix, iy = int(x), int(y)
                    rgb[ix, iy] += np.array([60, 60, 60], dtype=np.uint8)
        for node in self.nodes:
            x, y = int(node.x), int(node.y)
            par = rgb[x-2:x+3, y-2:y+3]
            if node.completed:
                add_color(par, np.array([1]), [0, 80, 0])
            elif node.locked:
                add_color(par, np.array([1]), [80, 80, 0])
            elif len(node.visiters)>0:
                add_color(par, np.array([1]), [100, 50, 0])
            else:    
                add_color(par, np.array([1]), [80, 0, 0])
        for loc in self.locations:
            x, y = loc
            x, y = int(x), int(y)
            add_color(rgb[x-2:x+3, y-2:y+3], np.array([1]), [0, 0, 80])
        imageio.imwrite(path, rgb)

    def add_candidates(self, p):
        self.candidates.append(p)

    def cluster_candidates(self, cluster_radius=5.):
        valid = [True for _ in self.candidates] 
        ret = []
        for i, p in enumerate(self.candidates):
            if valid[i]:
                cluster = []
                for j, q in enumerate(self.candidates):
                    if valid[j] and self.distance(self.Node(*p), self.Node(*q)) <= cluster_radius:
                        valid[j] = False
                        cluster.append(q)
                ret.append(p)
        self.candidates = ret
    
    def select_candidates(self, x, y):
        x, y = int(x), int(y)
        p = (x, y)
        ret = [q for q in self.candidates if self.distance(self.Node(*p), self.Node(*q)) <= 70 and self.distance(self.Node(*p), self.Node(*q)) >= 30 and self.no_collision(self.Node(*p), self.Node(*q))]
        ret = [(px + 0.5, py + 0.5) for px, py in ret]
        return ret
    
    def assign_candidates(self):
        for node in self.nodes:
            node.candidates = self.select_candidates(node.x, node.y)

    def no_collision(self, p, q):
        map = copy.deepcopy(self.map)
        map = (map == 1).astype(np.uint8)
        for a in range(self.num_agents):
            x, y = self.locations[a]
            x, y = int(x), int(y)
            map[x-2:x+3, y-2:y+3] = 0.
        x0, y0 = p.x, p.y
        x1, y1 = q.x, q.y
        dis = math.hypot(x1-x0, y1-y0)
        r = int(dis * 2)+1
        dx = (x1-x0)/r
        dy = (y1-y0)/r
        x, y = x0, y0
        for i in range(r):
            x = x + dx
            y = y + dy

            ix, iy = int(x), int(y)
            if map[ix, iy] == 1:
                return False
        return True
    
    def no_collision_with_line(self, p1, q1):
        def check_intersect(p1, q1, node, last_node):
            px, py = p1
            qx, qy = q1
            crs = 0
            # print("(%.3f, %.3f), (%.3f, %.3f)"%(px, py, qx, qy))
            dx_last, dy_last = last_node.x - px, last_node.y - py
            dx_now, dy_now = node.x - px, node.y - py
            dx, dy = qx - px, qy - py
            crs_last = dx_last * dy - dx * dy_last
            crs_now = dx_now * dy - dx * dy_now
            # print('(%.5f, %.5f) -> (%.5f, %.5f)'%(node.parent.x, node.parent.y, node.x, node.y), crs_last, crs_now)
            if crs_last > 0 and crs_now < 0:
                crs += 1
            if crs_last < 0 and crs_now > 0:
                crs += 1
            
            dx, dy = node.x - last_node.x, node.y - last_node.y
            dx_p, dy_p = px - last_node.x, py - last_node.y
            dx_q, dy_q = qx - last_node.x, qy - last_node.y
            crs_p = dx_p * dy - dx * dy_p
            crs_q = dx_q * dy - dx * dy_q
            # print('(%.5f, %.5f) -> (%.5f, %.5f)'%(node.parent.x, node.parent.y, node.x, node.y), crs_p, crs_q)
            if crs_p > 0 and crs_q < 0:
                crs += 1
            if crs_p < 0 and crs_q > 0:
                crs += 1
            if crs == 2:
                return False
            return True
        for node in self.nodes:
            if node.parent is not None and not check_intersect(p1, q1, node, node.parent):
                return False
        return True
    
    def remove_node(self, node):
        if not node.exist:
            return
        if node.parent is not None and node.parent.exist:
            p = node.parent
            p.childs = [x for x in p.childs if x.id != node.id]
        node.exist = False
        for child in node.childs:
            self.remove_node(child)

    def remove_wrong_nodes(self):
        # remove nodes in the obstacle
        map = copy.deepcopy(self.map)
        map = (map == 1).astype(np.uint8)
        for a in range(self.num_agents):
            x, y = self.locations[a]
            x, y = int(x), int(y)
            map[x-2:x+3, y-2:y+3] = 0.
        all_nodes = self.nodes
        self.nodes = []
        for node in all_nodes:
            x, y = int(node.x), int(node.y)
            if map[x-2:x+3, y-2:y+3].sum() > 0 and node.id != self.main_root.id:
                # for _ in range(10):
                #     # try add noise to find feasible cell
                # 
                self.remove_node(node)

            #if node.parent is not None and not self.no_collision(node, node.parent):
            #    self.remove_node(node)
        for node in all_nodes:
            if node.exist:
                self.nodes.append(node)
        for a in range(self.num_agents):
            if (not self.agent_state[a][1].exist) or (self.agent_state[a][0] == "to" and not self.agent_state[a][2].exist):
                node, _ = self.get_closest_node(self.Node(*self.locations[a]))
                assert node.exist
                self.agent_state[a] = ('to', node, node)

    '''def find_target(self, node, a):
        if not node.exist or (node.locked and node.keeper != a):
            return None
        random.shuffle(node.childs)
        for x in node.childs:
            tmp = self.find_target(x, a)
            if tmp is not None:
                return tmp
        # if len(node.visiters) == 0:
        if len(node.candidates) > 0:
            return node
        return None'''
    
    def find_target(self, node, a, father=None, subtree=True):
        if not node.exist or (node.locked and node.keeper != a):
            return None
        next_nodes = node.childs + [node.parent] if not subtree and node.parent is not None else []
        if father is not None:
            next_nodes = [x for x in next_nodes if x.id != father.id]
        for x in next_nodes:
            ret = self.find_target(x, a, father=node)
            if ret is not None:
                return ret
        if len(node.candidates) > 0:
            return node
        return None

    def move(self, rrt_expand = True):
        self.step += 1
        if self.stage == "FindMainRoot":
            print(self.stage)
            return [[self.main_root.x, self.main_root.y] for _ in range(self.num_agents)]
        print(self.stage, "{} nodes".format(len(self.nodes)))
        # expand rrt nodes
        rrt_expand = rrt_expand or not hasattr(self, 'candidates')
        map = copy.deepcopy(self.map) # self.map就是包含0，1，2的边界识别map
        unexplored = copy.deepcopy(self.unexplored)
        map = (map == 1).astype(np.int32) # 变成全obstacle的map，1为障碍，0为其他
        unexplored[map == 1] = 0 # 去除obstacle
        for a in range(self.num_agents):
            x, y = self.locations[a]
            x, y = int(x), int(y)
            map[x-2:x+3, y-2:y+3] = 0. # 小车附近区域认为无障碍
        
        if rrt_expand:
            for node in self.nodes:
                node.candidates.clear() # 清空list
            self.candidates = []
            for iter in range(self.max_iter):
                rx, ry = self.get_random_point() # 在大地图中随机取点
                node, dist = self.get_closest_node(self.Node(rx, ry)) # 找到距离这个点最近的在集合里的点
                if dist > self.node_radius: # 如果最近节点与随机点的距离大于设定的节点半径，则开始扩展新节点
                    # steer
                    dx = rx - node.x
                    dy = ry - node.y
                    r = ((dx ** 2) + (dy ** 2)) ** 0.5
                    dx /= r
                    dy /= r

                    x, y = node.x, node.y
                    fx, fy = int(x), int(y)
                    for t in range(self.expand_dis): # 往新生成的点的方向延展，只要延展到障碍区域就停下
                        x += dx
                        y += dy
                        ix, iy = int(x), int(y)
                        if map[ix-2:ix+3, iy-2:iy+3].sum() > 0: # or unexplored[ix, iy] == 1:
                            break
                        fx, fy = ix, iy
                    
                    if unexplored[fx, fy]:
                        self.add_candidates((fx, fy)) # 如果为被探索，则添加到候选节点
                        new = True
                    else: # 如果这个点是已经探索过的
                        # 检测一条线段是否与快速随机树（RRT）中的任何节点连接产生交叉（即碰撞）
                        if self.no_collision_with_line((fx+0.5, fy+0.5), node.position()):
                            new_node, new = self.add_node(loc=(fx+0.5, fy+0.5), parent=node)
                    # if new:
                    #     x = node
                    #     while x is not None:
                    #         x.completed=False
                    #         x = x.parent

            # self.evict_candidates()
            self.cluster_candidates(cluster_radius=self.node_radius) # 进行点的聚类，聚类半径是cluster_radius
            self.assign_candidates() # 将self.candidate的点分配到各个node的candidates

        for node in self.nodes:
            node.lock_count_down() # 解锁节点

        goals = []
        for a in range(self.num_agents):
            # 检查智能体 a 是否处于 'at'（在某个节点上）状态，或者处于 'to'（前往某个节点）状态但目标节点没有候选节点。
            if self.agent_state[a][0] == 'at' or (self.agent_state[a][0] == 'to' and len(self.agent_state[a][2].candidates) == 0):
                for node in self.nodes:
                    if node.keeper == a:
                        node.unlock()
                # find next node
                nodeA = self.agent_state[a][1] # nodeA表示智能体所在的node或者目标node
                nodeA.unlock()
                if a not in nodeA.visiters:
                    nodeA.visiters.append(a) # 将智能体 a 添加到节点的访问者列表中
                candidates = self.select_candidates(nodeA.x, nodeA.y)
                if self.agent_state[a][0] == 'at' and len(candidates) > 0:
                    self.agent_state[a] = ('try', nodeA, random.choice(candidates))
                    goals.append(self.agent_state[a][2])
                    continue
                '''all_candidates = [('virtual', (x, y), self.num_agents + 1) for x, y in candidates] + [('real', node, self.num_agents - len(node.visiters) + 1) for node in nodeA.childs if not node.completed and not node.locked]
                possible_nodes = [('real', node) for node in nodeA.childs if not node.completed]
                if len(all_candidates) == 0:
                    nodeA.completed = (len(possible_nodes) == 0)
                    if nodeA.parent is not None:
                        self.agent_state[a] = ('to', nodeA, nodeA.parent)
                        goals.append([self.agent_state[a][2].x, self.agent_state[a][2].y])
                    else:
                        self.agent_state[a] = ('at', nodeA, None)
                        goals.append([nodeA.x, nodeA.y])
                else:
                    prob = torch.tensor([c[2] for c in all_candidates])
                    print(prob)
                    prob = Categorical(prob/prob.sum())
                    idx = prob.sample().item()
                    w = all_candidates[idx] 
                    # w = random.choice(all_candidates)
                    if w[0] == 'virtual':
                        x, y = w[1]
                        self.agent_state[a] = ('try', nodeA, (x, y))
                        goals.append([x, y])
                    else:
                        nodeB = w[1]
                        nodeB.lock(lock_time=10)
                        self.agent_state[a] = ('to', nodeA, nodeB)
                        goals.append([self.agent_state[a][2].x, self.agent_state[a][2].y])'''
                now = nodeA # 如果找不到候选的节点则进行下面的步骤
                found = False
                while True:
                    if now is None: # now是none的时候说明已经到达父节点
                        target_node = self.find_target(nodeA, a, subtree=False)
                        break
                    else:
                        target_node = self.find_target(now, a, subtree=True)
                    if target_node is not None:
                        found = True
                        self.agent_state[a] = ('to', nodeA, target_node)
                        # add lock
                        u = nodeA
                        while u is not None and u.id != now.id:
                            u.lock(20, keeper=a)
                            u = u.parent
                        u = target_node
                        while u is not None and u.id != now.id:
                            u.lock(20, keeper=a)
                            u = u.parent
                        # do not lock the common ancester `now`
                        goals.append([target_node.x, target_node.y])
                        break
                    if self.strict:
                        found = True
                        if nodeA.parent is not None:
                            self.agent_state[a] = ("to", nodeA, nodeA.parent)
                        else:
                            self.agent_state[a] = ("to", nodeA, nodeA)
                        break
                    now = now.parent
                if not found:
                    if len(self.candidates) > 0: # 如果找不到好的结果就随机选点
                        self.agent_state[a] = ('try', nodeA, random.choice(self.candidates))
                        goals.append(self.agent_state[a][2])
                    else:
                        self.agent_state[a] = ('at', nodeA, None)
                        goals.append([nodeA.x, nodeA.y])
            elif self.agent_state[a][0] == 'to':
                goals.append([self.agent_state[a][2].x, self.agent_state[a][2].y])
            elif self.agent_state[a][0] == 'try':
                locB = self.agent_state[a][2]
                if map[int(locB[0]), int(locB[1])] == 1 or unexplored[int(locB[0]), int(locB[1])] == 0: # or not self.no_collision(self.agent_state[a][1], self.Node(*locB)):
                    self.agent_state[a] = ('to', self.agent_state[a][1], self.agent_state[a][1])
                    goals.append([self.agent_state[a][2].x, self.agent_state[a][2].y])
                else:
                    goals.append(locB)
            print("move")
            loc1 = self.locations[a]
            if self.agent_state[a][0] == "try":
                print(f"agent {a}, loc {loc1}, state {self.agent_state[a][0]} {self.agent_state[a][2]}")
            elif self.agent_state[a][0] == "to":
                print(f"agent {a},  loc {loc1}, state {self.agent_state[a][0]} {self.agent_state[a][2].position()}")
            elif self.agent_state[a][0] == "at":
                print(f"agent {a},  loc {loc1}, state {self.agent_state[a][0]} {self.agent_state[a][1].position()}")
        self.main_root.unlock()
        assert len(goals) == self.num_agents
        return goals

    def get_random_point(self):
        return (random.uniform(self.lx, self.rx), random.uniform(self.ly, self.ry))


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 6, 6),
        (3, 6, 7, 8),
        (3, 8, 5, 9),
        (3, 5, 13, 6),
        (7, 5, 8, 10),
        (9, 5, 10, 6),
        (1, 6, 3, 12),
        (10, 7, 11, 14),
        (3, 10, 5, 14),
        (7, 2, 8, 6),
        (9, 3, 16, 4),
        (5, 12, 10, 14)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt = RRT(
        start=[6.1,10.1],
        goals=[[0.5,2.5]],
        rand_area=[[-2, 14], [0, 14]],
        obstacle_list=obstacle_list,
        expand_dis=0.5 ,
            goal_sample_rate=0,
                 max_iter=10000)
    direction_k = 8
    unit = np.pi * 2 / direction_k
    sections = [(unit * i - unit/2, unit * i +  unit/2) for i in range(direction_k)]
    rrt.add_direction(sections[0])
    # Set Initial parameters
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    show_path = True
    # Draw final path
    if show_path:
        rrt.draw_graph()
        if type(path)!= type(None):
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
        plt.grid(True)
        plt.savefig('/home/gaojiaxuan/onpolicy/onpolicy/scripts/gjx_tmp/rrt.png')

    fig, ax = plt.subplots(1, 2, figsize = (6*16/9, 6), facecolor="whitesmoke", num="test")
    ax = [ax]
    for i in range(1):
        for j in range(2):
            ax[i][j].clear()
            ax[i][j].set_yticks([])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xticklabels([])
    rrt.draw_graph(fig=fig, ax=ax[0][0])
    rrt.draw_graph(fig=fig, ax=ax[0][1])
    fig.savefig('/home/gaojiaxuan/onpolicy/onpolicy/scripts/gjx_tmp/rrt.png')


if __name__ == '__main__':
    main()