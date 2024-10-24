import numpy as np
from heapq import heappop, heappush
from math import sqrt
import pyautogui
import time

# 加载障碍物网格文件
obstacle_map = np.loadtxt('./map_grid.txt', dtype=int)

# 地图和网格大小
GRID_SIZE = 70
CELL_SIZE = 5

# 初始化按键暂停时间
pyautogui.PAUSE = 0

# 全局变量
g_center = None
g_center_cache = None
g_center_last_update_time = 0
G_CENTER_CACHE_DURATION = 0.5  # 缓存有效期，单位秒
G_CENTER_MISS_THRESHOLD = 5  # 连续未检测到 g_center 的阈值

# 初始化按键状态
key_status = {'w': False, 'a': False, 's': False, 'd': False}

# 添加优先跟随的英雄列表
priority_heroes = ["敖隐", "莱西奥", "戈娅", "艾琳", "蒙犽", "伽罗", "公孙离", "黄忠", "成吉思汗", "虞姬", "李元芳", "后羿", "狄仁杰", "马可波罗", "鲁班七号", "孙尚香"]

# 加载真实类别名称
with open(r'./name.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    class_names = {}
    for line in lines:
        line = line.strip()
        if line and ':' in line and not line.startswith('#'):
            key, value = line.split(':', 1)
            try:
                key = int(key)
                class_names[key] = value.strip().strip("'")
            except ValueError:
                continue

class Node:
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def heuristic_chebyshev(a, b):
    D, D2 = 1, sqrt(2)
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def a_star(start, goal, obstacle_map):
    open_set = []
    heappush(open_set, (0, Node(start[0], start[1], 0)))
    closed_set = set()
    g_score = {start: 0}

    while open_set:
        current_node = heappop(open_set)[1]
        current = (current_node.x, current_node.y)

        if current == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and obstacle_map[neighbor[1], neighbor[0]] == 0:
                move_cost = sqrt(2) if dx != 0 and dy != 0 else 1
                tentative_g_score = g_score[current] + move_cost
                if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic_chebyshev(neighbor, goal)
                    heappush(open_set, (f_score, Node(neighbor[0], neighbor[1], f_score, current_node)))
    return None

def convert_to_grid_coordinates(pixel_x, pixel_y):
    return int(pixel_x // CELL_SIZE), int(pixel_y // CELL_SIZE)

def handle_key(key, action):
    if action == 'press' and not key_status[key]:
        pyautogui.keyDown(key)
        key_status[key] = True
    elif action == 'release' and key_status[key]:
        pyautogui.keyUp(key)
        key_status[key] = False

def release_all_keys():
    for key in key_status:
        handle_key(key, 'release')

def move_direction(dx, dy):
    current_keys = {'w': False, 'a': False, 's': False, 'd': False}
    abs_dx, abs_dy = abs(dx), abs(dy)

    if abs_dx > abs_dy:
        current_keys['d'] = dx > 0
        current_keys['a'] = dx < 0
    else:
        current_keys['s'] = dy > 0
        current_keys['w'] = dy < 0

    for key, is_pressed in current_keys.items():
        handle_key(key, 'press' if is_pressed else 'release')

    print(f"移动方向: dx={dx}, dy={dy}, 按键状态: {current_keys}")

def find_priority_target(b_centers, g_center):
    priority_targets = []
    closest_target = None
    min_distance = float('inf')

    for b_center in b_centers:
        distance = sqrt((g_center[0] - b_center[0]) ** 2 + (g_center[1] - b_center[1]) ** 2)
        class_id = b_center[2]
        hero_name = class_names.get(class_id, "未知英雄")

        if hero_name in priority_heroes:
            priority_targets.append((b_center, distance))
        elif not priority_targets and (closest_target is None or distance < min_distance):
            closest_target = b_center
            min_distance = distance

    if priority_targets:
        target, _ = min(priority_targets, key=lambda x: x[1])
    else:
        target = closest_target

    return target

def model1_movement_logic(detection_result):
    global g_center, g_center_cache, g_center_last_update_time
    g_center_miss_count = 0

    g_center = detection_result['g_center']
    b_centers = detection_result['b_centers']

    print(f"模态1 A* 处理检测结果: g_center={g_center}, b_centers数量={len(b_centers)}")

    current_time = time.time()

    if g_center:
        g_center_cache = g_center
        g_center_last_update_time = current_time
        g_center_miss_count = 0
    elif g_center_cache and (current_time - g_center_last_update_time) < G_CENTER_CACHE_DURATION:
        g_center = g_center_cache
        print(f"[{current_time}] 使用缓存的 g_center 位置: {g_center}")
        g_center_miss_count += 1
    else:
        g_center_miss_count += 1
        if g_center_miss_count >= G_CENTER_MISS_THRESHOLD:
            print(f"[{current_time}] g_center 连续 {G_CENTER_MISS_THRESHOLD} 次未检测到")
            release_all_keys()
            return {'g_center': None, 'closest_b': None, 'is_moving': False}
        else:
            print(f"[{current_time}] g_center 暂时未检测到，继续使用上一个有效位置")
            g_center = g_center_cache

    if g_center and b_centers:
        target = find_priority_target(b_centers, g_center)
        if target:
            g_grid = convert_to_grid_coordinates(g_center[0], g_center[1])
            b_grid = convert_to_grid_coordinates(target[0], target[1])
            path = a_star(g_grid, b_grid, obstacle_map)
            if path and len(path) > 1:
                next_step = path[1]
                world_x, world_y = next_step[0] * CELL_SIZE + CELL_SIZE // 2, next_step[1] * CELL_SIZE + CELL_SIZE // 2
                dx, dy = world_x - g_center[0], world_y - g_center[1]
                move_direction(dx, dy)
                print(f"模态1 A* 移动，路径长度: {len(path)}, 下一步: {next_step}")
                return {'g_center': g_center, 'closest_b': target, 'is_moving': True}
            else:
                # A* 没有找到路径，直接朝目标移动
                dx, dy = target[0] - g_center[0], target[1] - g_center[1]
                move_direction(dx, dy)
                print(f"模态1 直接移动，目标位置: {target}")
                return {'g_center': g_center, 'closest_b': target, 'is_moving': True}
        else:
            print("模态1 A* 未找到目标")
    else:
        print("模态1 A* 没有检测到 g_center 或 b_centers")

    release_all_keys()
    return {'g_center': g_center, 'closest_b': None, 'is_moving': False}

if __name__ == "__main__":
    # 这里可以添加测试代码
    pass
