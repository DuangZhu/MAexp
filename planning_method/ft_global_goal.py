# References:
# MAANS: https://github.com/zoeyuchao/maans
# --------------------------------------------------------
import numpy as np
from queue import deque
import torch
from planning_method.RRT.rrt import RRT, WMA_RRT


def ft_get_goal(self, inputs, goal_mask, pre_goals = None):
    obstacle = inputs['map_pred']
    explored = inputs['exp_pred']
    locations = inputs['locations']

    if all(goal_mask):
        return pre_goals
    
    # start = time.time()
    obstacle = np.rint(obstacle).astype(np.int32)
    explored = np.rint(explored).astype(np.int32)
    explored[obstacle == 1] = 1

    H, W = explored.shape
    steps = [(-1,0),(1,0),(0,-1),(0,1)]
    map, (lx, ly), unexplored = get_frontier(obstacle, explored, locations, cut_boundary=(self.config['algo'] != "ft_wma_rrt"))
    '''
    map: H x W
        - 0 for explored & available cell
        - 1 for obstacle
        - 2 for target (frontier)
    '''
    goals = []
    locations = [(x-lx, y-ly) for x, y in locations] 
    if self.config['algo'] in ['ft_wma_rrt']:
        if not hasattr(self, "wma_rrt") or self.infos['local_Steps'] == 0:
            self.wma_rrt = WMA_RRT(
                start=locations,
                rand_area=((0, H), (0, W)),
                expand_dis=30,
                node_radius=4,
                max_iter=2000,
                map=map, 
                unexplored=unexplored,
                strict=True)
        self.wma_rrt.update_map(map, unexplored, locations)
        goals = self.wma_rrt.move(rrt_expand=(self.infos['local_Steps'] % 5 == 0))
        goals = np.array([(int(x), int(y)) for x, y in goals])
        print(goals)
    elif self.config['algo'] in ['ft_utility', 'ft_voronoi']:
        pre_goals = pre_goals.copy()
        pre_goals[:, 0] -= lx 
        pre_goals[:, 1] -= ly
        if self.config['algo'] == 'ft_utility':
            goals = max_utility_frontier(map, unexplored, locations, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, utility_radius = self.all_args.utility_radius, pre_goals = pre_goals, goal_mask = goal_mask, random_goal=self.all_args.ft_use_random)
        elif self.config['algo'] == 'ft_voronoi':
            goals = voronoi_based_planning(map, unexplored, locations, clear_radius = self.config['ft_para']['clear_radius'], cluster_radius = self.config['ft_para']['cluster_radius'], utility_radius = self.config['ft_para']['utility_radius'], pre_goals = pre_goals, goal_mask = goal_mask, random_goal=self.config['ft_para']['random_goal'])
        goals[:, 0] += lx
        goals[:, 1] += ly
    else:
        for agent_id in range(self.config['num_agent']):
            if goal_mask[agent_id]:
                goals.append((-1,-1))
                continue
            if self.config['algo'] == 'ft_apf':
                apf = APF(self.config)
                path = apf.schedule(map, unexplored, locations, steps, agent_id, clear_disk = self.config['ft_para']['clear_disk'], random_goal=self.config['ft_para']['random_goal']) # random
                goal = path[-1]             
            elif self.config['algo'] == 'ft_nearest':
                goal = nearest_frontier(map, unexplored, locations, steps, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, random_goal=self.all_args.ft_use_random)
            elif self.config['algo'] == 'ft_rrt':
                goal = rrt_global_plan(map, unexplored, locations, agent_id, clear_radius = self.config['ft_para']['clear_radius'], cluster_radius = self.config['ft_para']['cluster_radius'], utility_radius = self.config['ft_para']['utility_radius'], random_goal=self.config['ft_para']['random_goal'], expand_dis=self.config['ft_para']['expand_dis']) # random
            else:
                raise NotImplementedError
            goals.append((goal[0] + lx, goal[1] + ly))
        goals = np.array(goals)
    # end = time.time()
    # print(str(end-start))
    return goals

def l2distance(a,b):
    return pow(pow(a[0]-b[0],2)+pow(a[1]-b[1],2),0.5)

def add_clear_disk(map, unexplored, r, loc):
    map = map.copy()
    unexplored = unexplored.copy()
    map[map==2] = 0
    map[unexplored == 1] = 3 
    H, W = map.shape
    for x in range(H):
        for y in range(W):
            if l2distance((x,y), loc) <= r and map[x,y] == 3: 
                map[x,y] = 0 
                unexplored[x,y] = 0 
    steps = [(0,1),(0,-1),(1,0),(-1,0)]
    for x in range(H):
        for y in range(W):
            if map[x,y] == 0:
                neighbors = [(x+dx, y+dy) for dx, dy in steps if 0 <= x+dx < H and 0 <= y+dy < W]
                if sum([(unexplored[u,v]==1) for u,v in neighbors])>0 and sum([map[u,v] == 1 for u,v in neighbors]) == 0: 
                    map[x,y] = 2 
    unexplored = (map == 3).astype(np.uint8) 
    map[unexplored == 1] = 0 
    return map, unexplored

def get_boundary(map, v):
    row = ((map==v).astype(np.int32).sum(1)>0).astype(np.int32).tolist()
    if 1 not in row:
        return map.shape[0], 0, map.shape[1], 0
    x1 = row.index(1)
    x2 = map.shape[0] - 1 - list(reversed(row)).index(1)

    col = ((map==v).astype(np.int32).sum(0)>0).astype(np.int32).tolist()
    y1 = col.index(1)
    y2 = map.shape[1] - 1 - list(reversed(col)).index(1)

    return x1, x2, y1, y2

def get_frontier(obstacle, explored, locations, cut_boundary=True):
    explored[obstacle == 1] = 1
    H, W = explored.shape
    steps = [(-1,0),(1,0),(0,-1),(0,1)]
    map = np.ones((H, W)).astype(np.int32) * 3 
    map[explored == 1] = 0
    map[obstacle == 1] = 1 
    num_agents = len(locations)
    # frontier
    lx, rx, ly, ry = 1e9, 0, 1e9, 0
    # boundary
    map[0, :] = 1
    map[H-1, :] = 1
    map[:, 0] = 1
    map[:, W-1] = 1
    que = deque([(1,1)])
    pad = 10
    
    def fix_boundary(lx, rx, ly, ry, H, W):    
        lx = max(lx, 0)
        rx = min(rx, H-1)
        ly = max(ly, 0)
        ry = min(ry, W-1)
        return lx, rx, ly, ry
    x1, x2, y1, y2 = get_boundary(map, 0) 
    lx = min(lx, x1)
    rx = max(rx, x2)
    ly = min(ly, y1)
    ry = max(ry, y2)
    x1, x2, y1, y2 = get_boundary(explored, 1)
    lx = min(lx, x1 - pad)
    rx = max(rx, x2 + pad)
    ly = min(ly, y1 - pad)
    ry = max(ry, y2 + pad)
    for agent_id in range(num_agents):
        lx = min(lx, locations[agent_id][0]-pad)
        ly = min(ly, locations[agent_id][1]-pad)
        rx = max(rx, locations[agent_id][0]+pad)
        ry = max(ry, locations[agent_id][1]+pad)
    lx, rx, ly, ry = fix_boundary(lx, rx, ly, ry, H, W)
    for x in range(lx, rx+1):
        for y in range(ly, ry+1):
            if map[x,y] == 0:
                neighbors = [(x+dx, y+dy) for dx, dy in steps]
                if sum([(map[u,v] == 3) for u,v in neighbors])>0 and sum([map[u,v] == 1 for u,v in neighbors]) == 0: # neighbors are unexplored and no walls
                    map[x,y] = 2 
    unexplored = (map == 3).astype(np.int8)
    map[map == 3] = 0 

    x0, y0 = int((lx+rx)/2), int((ly+ry)/2)
    
    lx = min(lx, x0-40)
    rx = max(rx, x0+40)
    ly = min(ly, y0-40)
    ry = max(ry, y0+40)
    
    lx, rx, ly, ry = fix_boundary(lx, rx, ly, ry, H, W)

    unexplored[map == 1] = 0
    if not cut_boundary:
        lx = ly = 0
        rx, ry = H-1, W-1
    else:
        map[lx, :] = 1
        map[rx, :] = 1
        map[:, ly] = 1
        map[:, ry] = 1
    return map[lx:rx+1, ly:ry+1], (lx, ly), unexplored[lx:rx+1, ly:ry+1]

def get_frontier_cluster(frontiers, cluster_radius = 5.0, cluster_size = 5):
    if len(frontiers) == 0:
        return []
    num_frontier = len(frontiers)
    clusters = []
    H = max([x for (x,y) in frontiers]) + 1
    W = max([y for (x,y) in frontiers]) + 1
    valid = np.zeros((H,W), dtype=np.uint8)
    for x,y in frontiers:
        valid[x,y] = 1
    cluster_radius = int(cluster_radius)
    for i in range(num_frontier):
        if valid[frontiers[i][0], frontiers[i][1]] == 1:
            neigh = []
            lx, rx, ly, ry = frontiers[i][0]-cluster_radius, frontiers[i][0]+cluster_radius, frontiers[i][1]-cluster_radius, frontiers[i][1]+cluster_radius
            lx = max(lx, 0)
            rx = min(rx, H-1)
            ly = max(ly, 0)
            ry = min(ry, W-1)
            for x in range(lx, rx+1):
                for y in range(ly, ry+1):
                    if valid[x,y] == 1:
                        valid[x,y] = 0
                        neigh.append((x,y))
            center = None
            min_r = 1e9
            for p in neigh:
                r = max([l2distance(p,q) for q in neigh])
                if r<min_r:
                    min_r = r
                    center = p
            if len(neigh) >= cluster_size:
                clusters.append({'center': center, 'weight': len(neigh)})
    return clusters

def nearest_frontier(map, unexplored, locations, steps, agent_id, clear_radius = 40, cluster_radius = 5, random_goal = False):
    map, unexplored = add_clear_disk(map, unexplored, clear_radius, locations[agent_id])
    H, W = map.shape
    que = deque([locations[agent_id]])
    vis = np.zeros((H, W), dtype=np.int8)
    dis = np.zeros((H, W), dtype=np.int32)
    vis[locations[agent_id][0], locations[agent_id][1]] = 1
    while len(que)>0:
        x, y = que.popleft()
        neighbors = [(x+dx, y+dy) for dx, dy in steps]
        for u,v in neighbors:
            if map[u,v] in [0,2] and vis[u,v] == 0:
                vis[u,v] = 1
                dis[u,v] = dis[x,y] + 1
                que.append((u,v))
    min_dis = 1e9
    min_x, min_y = None, None
    frontiers = []
    for x in range(H):
        for y in range(W):
            if map[x,y] == 2 and l2distance((x,y), locations[agent_id]) > clear_radius:
                frontiers.append((x,y))
    clusters = get_frontier_cluster(frontiers, cluster_radius=cluster_radius)
    for cluster in clusters:
        p = cluster['center']
        d = l2distance(p, locations[agent_id])
        if d > clear_radius:
            if min_dis > dis[p[0], p[1]]:
                min_x, min_y = p
                min_dis = dis[p[0], p[1]]
    if min_x == None:
        # no valid target
        if random_goal:
            x, y = np.random.randint(0, H), np.random.randint(0, W)
            while map[x,y] == 1:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
            min_x, min_y = x, y
        else:
            min_x, min_y = locations[agent_id][0], locations[agent_id][1]
    return min_x, min_y

def circle_matrix(H, W, p, radius):
    mat = np.zeros((H, W), dtype = np.int32)
    for x in range(H):
        for y in range(W):
            if l2distance((x,y), p)<= radius:
                mat[x,y] = 1
    return mat

def bfs_distance(map, lx, ly, start, goals):
    if lx == None:
        return 1e6
    H, W = map.shape
    sx, sy = start[0], start[1]
    sx -= lx
    sy -= ly
    '''if np.array(goals).size == 2:
        goals = [goals]'''
    if goals is not None:
        goals = [(max(0,min(x - lx, H-1)), max(0, min(y - ly, W-1)) ) for x, y in goals]
        num_goals = len(goals)
    # print(gx, gy)
    if sx < 0 or sy < 0 or sx >= H or sy >= W:
        return 1e6

    dis = np.zeros((H, W), dtype = np.int32)
    dis[sx, sy] = 1
    steps = [(-1,0),(1,0),(0,-1),(0,1)]

    que = deque([(sx, sy)])
    while len(que)>0:
        x, y = que.popleft()
        neighbors = [(x+dx, y+dy) for dx, dy in steps]
        neighbors = [(x, y) for x,y in neighbors if x>=0 and x<H and y>=0 and y<W]
        for u,v in neighbors:
            if map[u,v] in [0,2] and dis[u,v] == 0:
                dis[u,v] = dis[x,y] + 1
                que.append((u,v))
    dis[dis == 0] = 1e6
    if goals is None:
        return dis
    for i, (gx, gy) in enumerate(goals):
        if map[gx,gy] == 1:
            tar = (gx, gy)
            min_dis = 1e9
            for x in range(H):
                for y in range(W):
                    if map[x,y] in [0,2]:
                        tmp = l2distance((x,y), (gx, gy))
                        if tmp<min_dis:
                            min_dis = tmp
                            tar = (x,y)
            goals[i] = tar
    ret = [dis[gx, gy]-1 for gx, gy in goals]
    if num_goals == 1:
        return ret[0]
    return np.array(ret)

def max_utility_frontier(map, unexplored, locations,  clear_radius = 40, cluster_radius = 5, utility_radius = 50, pre_goals = None, goal_mask = None, random_goal = False):
    H, W = map.shape
    num_agents = len(locations)

    order = np.arange(num_agents)
    np.random.shuffle(order)
    unexplored = unexplored.copy().astype(np.int32)
    goals = np.zeros((num_agents, 2), dtype = np.int32)

    # masked agents
    if goal_mask == None:
        goal_mask = [False for _ in range(num_agents)]
    else:
        for agent_id in range(num_agents):
            if goal_mask[agent_id]:
                mat = circle_matrix(H, W, pre_goals[agent_id], utility_radius)
                unexplored[mat == 1] = 0


    o_map = map.copy()
    o_unexplored = unexplored.copy()
    for agent_id in order:
        if goal_mask[agent_id]:
            goals[agent_id] = pre_goals[agent_id]
            continue
        map, unexplored = add_clear_disk(map, unexplored, clear_radius, locations[agent_id])
        frontiers = []
        for x in range(H):
            for y in range(W):
                if map[x,y] == 2:
                    frontiers.append((x,y))
        clusters = get_frontier_cluster(frontiers, cluster_radius = cluster_radius)
        num_clusters = len(clusters)
        # compute utility
        max_utility = -1.0
        tar = None
        for it, cluster in enumerate(clusters):
            p = cluster['center']

            if max([l2distance(p, locations[i]) for i in range(num_agents)]) <= clear_radius:
                continue

            mat = circle_matrix(H, W, p, utility_radius)
            
            tmp = unexplored[mat == 1].sum()
            if tmp>max_utility:
                max_utility = tmp
                tar = p
        if tar == None:
            if random_goal:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
                while map[x,y] == 1:
                    x, y = np.random.randint(0, H), np.random.randint(0, W)
                tar = (x,y)
            else:
                tar = (locations[agent_id][0], locations[agent_id][1])
        goals[agent_id] = np.array(tar)
        mat = circle_matrix(H, W, tar, utility_radius)
        unexplored = o_unexplored
        unexplored[mat == 1] = 0
    # re-allocate goals ?
    dist = np.zeros((num_agents, num_agents), dtype=np.int32)
    for i in range(num_agents):
        dist[i] = bfs_distance(map, 0, 0, locations[agent_id], goals)
    tar = np.arange(num_agents)
    for T in range(10):
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                if (goal_mask[i] or l2distance(locations[i], goals[tar[j]]) > clear_radius) and (goal_mask[j] or l2distance(locations[j], goals[tar[i]]) > clear_radius) and dist[i, tar[j]] < dist[i, tar[i]] and dist[j,tar[i]] < dist[j,tar[j]]:
                    tmp = tar[i]
                    tar[i] = tar[j]
                    tar[j] = tmp
    ret = goals.copy()
    for i in range(num_agents):
        ret[i] = goals[tar[i]]
    return ret


def voronoi_based_planning(map, unexplored, locations, clear_radius = 40, cluster_radius = 5, utility_radius = 50, pre_goals = None, goal_mask = None, random_goal = False):
    H, W = map.shape
    num_agents = len(locations)

    o_unexplored = unexplored.copy()
    feasible = unexplored.copy()
    goals = np.zeros((num_agents, 2), dtype = np.int32)

    # masked agents
    if goal_mask == None:
        goal_mask = [False for _ in range(num_agents)]
    else:
        for agent_id in range(num_agents):
            if goal_mask[agent_id]:
                mat = circle_matrix(H, W, pre_goals[agent_id], utility_radius)
                feasible[mat == 1] = 0

    dist = np.zeros((num_agents, H, W), dtype=np.float32)
    o_map = map.copy()
    for i in range(num_agents):
        dist[i] = bfs_distance(map, 0, 0, locations[i], None)
    for agent_id in range(num_agents):
        if goal_mask[agent_id]:
            goals[agent_id] = pre_goals[agent_id]
            continue
        my_grids = np.ones_like(unexplored)
        for j in range(num_agents):
            if agent_id != j:
                my_grids[dist[agent_id] >= dist[j]] = 0.

        frontiers = []
        for x in range(H):
            for y in range(W):
                if map[x,y] == 2 and my_grids[x, y] == 1:
                    frontiers.append((x,y))
        clusters = get_frontier_cluster(frontiers, cluster_radius = cluster_radius)
        num_clusters = len(clusters)
        if num_clusters == 0:
            tar = None
        else:
            # compute utility
            max_utility = -1e9
            tar = None
            centers = [cluster['center'] for cluster in clusters]
            distance = np.array([dist[i][x, y] for x, y in centers]) # bfs_distance(map, 0, 0, locations[agent_id], centers)
            distance = distance / max(distance)
            for it, p in enumerate(centers):

                if l2distance(p, locations[agent_id]) <= clear_radius:
                    continue

                mat = circle_matrix(H, W, p, utility_radius)
                
                tmp = feasible[mat == 1].sum() / (utility_radius ** 2) / np.pi
                tmp = tmp - distance[it] * 1.
                if tmp > max_utility:
                    max_utility = tmp
                    tar = p
        if tar == None:
            if random_goal:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
                while map[x,y] == 1:
                    x, y = np.random.randint(0, H), np.random.randint(0, W)
                tar = (x,y)
            else:
                tar = (locations[agent_id][0], locations[agent_id][1])
        goals[agent_id] = np.array(tar)
        # print(locations[agent_id], tar, l2distance(locations[agent_id], tar), unexplored.sum(), max_utility)
        mat = circle_matrix(H, W, tar, utility_radius)
        feasible = o_unexplored
        feasible[mat == 1.] = 0.

        '''import imageio
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 2][my_grids.cpu().numpy() == 1.] += 100
        rgb[:, :, 1][feasible == 1.] += 100
        mat = circle_matrix(H, W, tar, 3)
        rgb[:, :, 0][mat == 1.] += 100

        mat = circle_matrix(H, W, locations[agent_id], 3)
        for c in range(3):
            rgb[:, :, c][mat == 1.] += 100
        imageio.imwrite(f"gjx/debug_{agent_id}.png", rgb)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 2][my_grids.cpu().numpy() == 1.] += 100
        imageio.imwrite(f"gjx/grids_{agent_id}.png", rgb)'''

    dist = np.zeros((num_agents, num_agents), dtype=np.int32)
    for i in range(num_agents):
        dist[i] = bfs_distance(map, 0, 0, locations[agent_id], goals)
    tar = np.arange(num_agents)
    for T in range(10):
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                if (not goal_mask[i] and l2distance(locations[i], goals[tar[j]]) > clear_radius) and (not goal_mask[j] and l2distance(locations[j], goals[tar[i]]) > clear_radius) and dist[i, tar[j]] < dist[i, tar[i]] and dist[j,tar[i]] < dist[j,tar[j]]:
                    tmp = tar[i]
                    tar[i] = tar[j]
                    tar[j] = tmp
    ret = goals.copy()
    for i in range(num_agents):
        ret[i] = goals[tar[i]]

    return ret

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

def rrt_global_plan(map, unexplored, locations, agent_id, clear_radius = 40, cluster_radius = 5, utility_radius = 50, random_goal = False, sections = None, get_farthest = False, return_rrt=False, return_success = False, return_targets = False, rrt_iterations=2000,expand_dis=30.0):
    map, unexplored = add_clear_disk(map, unexplored, clear_radius, locations[agent_id])
    H, W = map.shape
    map = map.astype(np.int32)
    loc = (locations[agent_id][0], locations[agent_id][1])
    map[loc[0] - 2: loc[0] + 3, loc[1] - 2 : loc[1] + 3] = 0

    # greedily assemble obstacles into rectangles to reduce the number of obstacles
    obstacles = find_rectangle_obstacles(map)
    
    # print("num of obstacles", len(obstacles))

    rrt = RRT(start=(loc[0] + 0.5, loc[1] + 0.5),
        goals=[],
        rand_area=((0, H), (0, W)),
        obstacle_list=obstacles,
        expand_dis=expand_dis ,
        goal_sample_rate=-1,
        max_iter=rrt_iterations)
    rrt.set_obs(map, unexplored)
    mat = circle_matrix(H, W, loc, clear_radius)
    rrt_map = unexplored.copy().astype(np.int32)
    rrt_map[mat == 1] = 0
    all_targets, success = rrt.select_frontiers(rrt_map, num_targets = 100, get_farthest=get_farthest, sections = sections)

    if sections is None:
        all_targets = [all_targets]
    
    goals = []

    for targets in all_targets:
        clusters = get_frontier_cluster(targets, cluster_radius = cluster_radius, cluster_size = 1)

        if len(clusters) == 0:
            if random_goal:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
                while map[x,y] == 1:
                    x, y = np.random.randint(0, H), np.random.randint(0, W)
                goal = (x,y)
            else:
                goal = (loc[0], loc[1])
            goals.append(goal)
            continue

        for cluster in clusters:
            center = cluster['center']
            # navigation cost
            nav_cost = l2distance(center, loc)
            # information gain
            mat = circle_matrix(H, W, center, utility_radius)
            area = mat.sum()
            info_gain = rrt_map[mat == 1].sum()
            info_gain /= area
            cluster['info_gain'] = info_gain
            cluster['nav_cost'] = nav_cost
        D = max([cluster['nav_cost'] for cluster in clusters]) + 0.01
        goal = None
        mx = -1e9
        for cluster in clusters:
            cluster['nav_cost'] /= D
            cluster['utility'] = cluster['info_gain'] - 1.0 * cluster['nav_cost']
            if mx < cluster['utility']:
                mx = cluster['utility']
                goal = cluster['center']
        
        #if rrt_iterations>0:
        #    print("num of targets", len(targets))
        #    print("num of clusters", len(clusters))
        #    print("Max dist", D - 0.1)
        goals.append(goal)
    
    if sections is None:
        ret = [goals[0], ]
    else:
        ret = [goals, ]
    if return_targets:
        ret += [all_targets]
    if return_rrt:
        ret += [rrt]
    if return_success:
        ret += [success]
    return ret if len(ret)>1 else ret[0]

def dilate(a):
    map = (a == 1)
    H, W = map.shape
    ret = map.copy()
    ret[1:] = ret[1:] + map[:H-1]
    ret[:H-1] = ret[:H-1] + map[1:]
    ret[:, 1:] = ret[:, 1:] + map[:, :W-1]
    ret[:, :W-1] = ret[:, :W-1] + map[:, 1:]
    return ret.astype(np.int32)

def get_closest_frontier(map, start, goal):
    start = [int(start[0]), int(start[1])]
    goal = (int(goal[0]), int(goal[1]))

    obstacle = np.rint(map[0])
    explored = np.rint(map[1])

    if explored[goal[0], goal[1]] == 1:
        return goal

    H, W = explored.shape

    start[0] = max(0, min(start[0], H-1))
    start[1] = max(0, min(start[1], W-1))

    row = np.array([(i-start[0])**2 for i in range(H)]).repeat(W).reshape(H, W)
    col = np.array([(i-start[1])**2 for i in range(W)]).repeat(H).reshape(W, H).transpose()

    dist_map = row+col

    explored[dist_map <=50] = 1
    explored[explored > 1] = 1

    unexplored = (explored == 0).astype(np.int32)
    tmp = dilate(unexplored) * (1-obstacle)
    for _ in range(3):
        unexplored = tmp
        tmp = dilate(unexplored) * (1-obstacle)
    frontier_map = (tmp - unexplored) * (1 - obstacle)

    row = np.array([(i-goal[0])**2 for i in range(H)]).repeat(W).reshape(H, W)
    col = np.array([(i-goal[1])**2 for i in range(W)]).repeat(H).reshape(W, H).transpose()

    dist_map = row+col

    dist = dist_map * frontier_map + (1-frontier_map) * 1e9

    (index_a, index_b) = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    return (index_a, index_b)


class APF(object):
    def __init__(self, args):
        self.args = args

        self.cluster_radius = args['ft_para']['cluster_radius']
        self.k_attract = args['ft_para']['k_attract']            # attraction coefficient
        self.k_agents = args['ft_para']['k_agents']              # coefficient for inter‐agent interaction
        self.AGENT_INFERENCE_RADIUS = args['ft_para']['AGENT_INFERENCE_RADIUS']  # radius within which an agent perceives other agents
        self.num_iters = args['ft_para']['num_iters']            # maximum number of iterations
        self.repeat_penalty = args['ft_para']['repeat_penalty']  # penalty for revisiting the same location
        self.dis_type = args['ft_para']['dis_type']              # type of distance calculation ('l1' or 'l2')
        self.use_random = args['ft_para']['use_random']          # whether to use random target selection
        self.clear_radius = args['ft_para']['clear_radius']      # used to filter random target points (points whose distances fall between 2× and 4× are chosen), also used to expand the boundary
        self.num_agents = args['num_agent']
        self.num_clusters = 100


    def distance(self, a, b): 
        a = np.array(a)
        b = np.array(b)
        if self.dis_type == "l2":
            return np.sqrt(((a-b)**2).sum())
        elif self.dis_type == "l1":
            return abs(a-b).sum()

    def schedule(self, map, unexplored, locations, steps, agent_id, penalty = None, full_path = True, clear_disk = False, random_goal = False):
        '''
        APF to schedule path for agent agent_id
        map: H x W
            - 0 for explored & available cell
            - 1 for obstacle
            - 2 for target (frontier)
        locations: num_agents x 2
        steps: available actions
        penalty: repeat penalty
        full_path: default True, False for single step (i.e., next cell)
        '''
        H, W = map.shape
        
        # find available targets
        vis = np.zeros((H,W), dtype = np.uint8)
        que = deque([]) 
        x, y = locations[agent_id]
        locx, locy = x, y
        vis[x,y] = 1
        que.append((x,y))
        sdis = np.zeros((H, W), dtype = np.uint8)
        random_targets = []
        while len(que)>0:
            x, y = que.popleft() 
            for dx, dy in steps:
                x1 = x + dx
                y1 = y + dy
                if vis[x1,y1] == 0 and map[x1,y1] in [0,2]: 
                    vis[x1,y1] = 1 
                    sdis[x1, y1] = sdis[x,y] + 1 
                    que.append((x1, y1))
                    if sdis[x1,y1] > self.clear_radius*2 and sdis[x1,y1] <self.clear_radius *4:
                        random_targets.append((x1,y1)) 
        
        targets = []
        if clear_disk:
            map, unexplored = add_clear_disk(map, unexplored, self.clear_radius, (locx, locy))
            near = []
            max_dist = -1
            for i in range(H):
                for j in range(W):
                    if map[i,j] == 2 and vis[i,j] == 1: 
                        if  l2distance((i,j), (locx,locy))>self.clear_radius:
                            targets.append((i,j)) 
                        else:
                            near.append((i,j))
                            if self.distance((i,j), (locx,locy)) > max_dist:
                                max_dist = self.distance((i,j), (locx,locy))
        else:
            for i in range(H):
                for j in range(W):
                    if map[i,j] == 2 and vis[i,j] == 1:
                        targets.append((i,j))
        # print("Number of targets", len(targets))
        # clustering
        clusters = []
        num_targets = len(targets)
        valid = [True for _ in range(num_targets)]
        t_targets = []
        for i in range(num_targets):
            if valid[i]:
                # not clustered
                chosen_targets = []
                for j in range(num_targets):
                    if valid[j] and self.distance(targets[i], targets[j]) <= self.cluster_radius:
                        valid[j] = False
                        chosen_targets.append(targets[j])
                min_r = 1e9
                center = None
                for a in chosen_targets:
                    max_d = max([self.distance(a,b) for b in chosen_targets])
                    if max_d < min_r:
                        min_r = max_d
                        center = a
                if len(chosen_targets) >= 3: 
                    clusters.append({"center": center, "weight": len(chosen_targets)})
                    for t in chosen_targets:
                        t_targets.append(t)
        # targets = t_targets
        num_targets = len(targets)
        
        # potential
        num_clusters = len(clusters) # 聚类
        if num_clusters == 0:
            for t in targets:
                clusters.append({'center':t, 'weight':1})
            num_clusters = len(clusters)
        potential = np.zeros((H, W))
        potential[map == 1] = 1e9 

        for i in range(num_clusters):
            for j in range(num_clusters):
                if i<j and clusters[i]['weight']/(sdis[clusters[i]['center'][0]][clusters[i]['center'][1]]+1) < clusters[j]['weight']/(sdis[clusters[j]['center'][0]][clusters[j]['center'][1]]+1):
                    tmp = clusters[i]
                    clusters[i] = clusters[j]
                    clusters[j] = tmp
        if num_clusters > self.num_clusters:
            num_clusters = self.num_clusters
            clusters = clusters[:num_clusters]
        # print("num clusters:", num_clusters)

        # potential of targets & obstacles (wave-front dist)
        for cluster in clusters:
            sx, sy = cluster["center"]
            w = cluster["weight"]
            dis = np.ones((H, W), dtype=np.int64) * 1e9
            dis[sx, sy] = 0
            que = deque([(sx, sy)])
            while len(que) > 0:
                (x, y) = que.popleft()
                for dx, dy in steps:
                    x1 = x + dx
                    y1 = y + dy
                    if x1 >= H or y1 >= W:
                        continue
                    if dis[x1, y1] == 1e9 and map[x1, y1] in [0,2]:
                        dis[x1, y1] = dis[x, y]+1 # wave-front distance
                        que.append((x1,y1))
            dis[sx, sy] = 1e9
            dis = 1 / dis
            dis[sx, sy] = 0 
            potential[map != 1] -= dis[map != 1] * self.k_attract * w 
        # potential of agents
        for agent_loc in locations:
            if agent_loc != (locx, locy):
                for x in range(H):
                    for y in range(W):
                        d = self.distance(agent_loc, (x,y))
                        if d <= self.AGENT_INFERENCE_RADIUS:
                            potential[x,y] += self.k_agents * (self.AGENT_INFERENCE_RADIUS - d) 
        # manual penalty (repeat penalty, etc.)
        if type(penalty) != type(None):
            potential += penalty

        # print("    %d targets"%num_targets)

        # schedule path
        it = 1
        current_loc = locations[agent_id]
        current_potential = 1e4
        minDis2Target = 1e9
        path = [(current_loc[0], current_loc[1])]
        while it <= self.num_iters and minDis2Target > 1:
            it = it + 1
            potential[current_loc[0], current_loc[1]] += self.repeat_penalty
            # print("    it : {}, loc : ({},{}), potential: {}".format(it, current_loc[0], current_loc[1], potential[current_loc[0], current_loc[1]]))
            best_neigh = None
            min_potential = 1e9
            for dx, dy in steps: 
                neighbor_loc = (current_loc[0] + dx, current_loc[1] + dy)
                if map[neighbor_loc[0], neighbor_loc[1]] == 1: 
                    continue
                if min_potential > potential[neighbor_loc[0], neighbor_loc[1]]:
                    min_potential = potential[neighbor_loc[0], neighbor_loc[1]]
                    best_neigh = neighbor_loc
            if current_potential > min_potential:
                current_potential = min_potential
                current_loc = best_neigh
                path.append(best_neigh)
            for tar in targets:
                l = self.distance(current_loc, tar)
                if l == 0:
                    continue
                minDis2Target = min(minDis2Target, l)
                if l<=1:
                    path.append((tar[0], tar[1]))
                    break
            if not full_path and len(path)>1:
                return path[1] # next grid
        # print("    Iters %d, Goal (%d, %d)"%(it, path[-1][0], path[-1][1]))
        random_plan = False
        if minDis2Target > 1:
            # random_plan = True
            random_plan = (l2distance(locations[agent_id], path[-1]) <= self.clear_radius)
        for i in range(agent_id):
            if locations[i][0]==locations[agent_id][0] and locations[i][1]==locations[agent_id][1]:
                random_plan = True # two agents are at the same location, replan
        random_plan = random_plan and self.use_random
        if random_plan and random_goal:
            # if not reaching a frontier, randomly pick a traget as goal
            if len(random_targets) == 0:
                random_targets.append((np.random.randint(0,H), np.random.randint(0,W)))
            w = np.random.randint(0, len(random_targets))
            path = (locations[agent_id], random_targets[w])
            # print("    random plan, pick goal (%d, %d)"%(random_targets[w][0], random_targets[w][1]))
        return path