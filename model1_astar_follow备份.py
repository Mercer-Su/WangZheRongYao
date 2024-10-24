import numpy as np
import logging
from heapq import heappop, heappush
from math import sqrt
from ultralytics import YOLO
import cv2
import mss
import pyautogui
import time
import threading
from threading import Event
from PIL import Image, ImageDraw, ImageFont

# 加载障碍物网格文件
obstacle_map = np.loadtxt('map_grid.txt', dtype=int)

# 地图和网格大小
GRID_SIZE = 70
CELL_SIZE = 5

# 初始化按键暂停时间
pyautogui.PAUSE = 0

# 设置日志级别为 ERROR
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# 加载YOLO模型
model = YOLO(r'best_perfect.pt')

# 捕获屏幕区域设置
sct = mss.mss()
monitor = {"top": 32, "left": 0, "width": 350, "height": 350}

# 帧率设置
fps = 30
frame_time = 1 / fps

# 全局变量
latest_path = []
g_center = None
g_center_cache = None
g_center_last_update_time = 0
G_CENTER_CACHE_DURATION = 0.5  # 缓存有效期，单位秒
G_CENTER_MISS_THRESHOLD = 7  # 连续未检测到 g_center 的阈值

# 线程控制变量
terminate_event = Event()
update_event = Event()
path_lock = threading.Lock()
g_center_lock = threading.Lock()

# 初始化按键状态
key_status = {'w': False, 'a': False, 's': False, 'd': False}

# 加载中文字体
font_path = r"C:\Windows\Fonts\simhei.ttf"  # 请确保这个路径是正确的
font = ImageFont.truetype(font_path, 20)

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

#置信度
confidence_threshold = 0.80

# 添加优先跟随的英雄列表
priority_heroes = ["敖隐", "莱西奥", "戈娅", "艾琳", "蒙犽", "伽罗", "公孙离", "黄忠", "成吉思汗", "虞姬", "李元芳", "后羿", "狄仁杰", "马可波罗", "鲁班七号", "孙尚香"]

def cv2_add_chinese_text(img, text, position, text_color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_color(class_id):
    if 0 <= class_id <= 122:
        return (0, 255, 0)  # 绿色
    elif 123 <= class_id <= 245:
        return (255, 0, 0)  # 蓝色
    else:
        return (0, 0, 255)  # 红色

class Node:
    """A* 算法节点类"""
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost


def heuristic_chebyshev(a, b):
    """启发式函数：切比雪夫距离"""
    D, D2 = 1, sqrt(2)
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def a_star(start, goal, obstacle_map):
    """A* 路径规划算法"""
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


def calculate_distance(x1, y1, x2, y2):
    """计算两点间的欧几里得距离"""
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def convert_to_grid_coordinates(pixel_x, pixel_y):
    """将像素坐标转换为网格坐标"""
    return int(pixel_x // CELL_SIZE), int(pixel_y // CELL_SIZE)


def handle_key(key, action):
    """按键处理函数"""
    if action == 'press' and not key_status[key]:
        pyautogui.keyDown(key)
        key_status[key] = True
    elif action == 'release' and key_status[key]:
        pyautogui.keyUp(key)
        key_status[key] = False


def release_all_keys():
    """释放所有按键"""
    for key in key_status:
        handle_key(key, 'release')


def move_direction(dx, dy):
    """根据 dx 和 dy 方向移动"""
    current_keys = {'w': False, 'a': False, 's': False, 'd': False}
    abs_dx, abs_dy = abs(dx), abs(dy)

    if abs_dx == abs_dy or abs(abs_dx - abs_dy) <= 5:
        diagonal_movement(dx, dy, current_keys)
    else:
        if abs_dx > abs_dy:
            current_keys['d'] = dx > 0
            current_keys['a'] = dx < 0
        else:
            current_keys['s'] = dy > 0
            current_keys['w'] = dy < 0

    for key, is_pressed in current_keys.items():
        handle_key(key, 'press' if is_pressed else 'release')


def diagonal_movement(dx, dy, current_keys):
    """处理对角线移动"""
    if dx > 0 and dy > 0:
        current_keys['d'], current_keys['s'] = True, True
    elif dx > 0 and dy < 0:
        current_keys['d'], current_keys['w'] = True, True
    elif dx < 0 and dy > 0:
        current_keys['a'], current_keys['s'] = True, True
    elif dx < 0 and dy < 0:
        current_keys['a'], current_keys['w'] = True, True


# 绘制路径函数
def draw_path_on_image(image, path, color=(0, 255, 0), thickness=2):
    """在图像上绘制路径"""
    with path_lock:
        for i in range(1, len(path)):
            start = (path[i - 1][0] * CELL_SIZE + CELL_SIZE // 2, path[i - 1][1] * CELL_SIZE + CELL_SIZE // 2)
            end = (path[i][0] * CELL_SIZE + CELL_SIZE // 2, path[i][1] * CELL_SIZE + CELL_SIZE // 2)
            cv2.line(image, start, end, color, thickness)


def check_and_clear_path():
    """检查并清空路径"""
    global latest_path
    with path_lock:
        if g_center is None or not latest_path:
            latest_path = []
            release_all_keys()


def update_g_center():
    """更新 g_center 的位置"""
    global g_center, g_center_cache, g_center_last_update_time
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
    results = model.track(frame, persist=True)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for i, box in enumerate(boxes):
            if 0 <= class_ids[i] <= 122:  # 对应原来的 "g" 类别
                with g_center_lock:
                    g_center = (box[0], box[1])
                    g_center_cache = g_center
                    g_center_last_update_time = time.time()
                print(f"[{time.time()}] g_center 更新: {g_center}")
                return True
    
    print(f"[{time.time()}] 未检测到 'g' 对象")
    return False


def start_move_along_path(new_path):
    """封装通用路径操作"""
    global latest_path
    with path_lock:
        latest_path = new_path
        update_event.set()

def find_priority_target(b_centers, g_center):
    """查找优先目标或最近的 b 类目标"""
    priority_targets = []
    closest_target = None
    min_distance = float('inf')

    for b_center in b_centers:
        distance = sqrt((g_center[0] - b_center[0]) ** 2 + (g_center[1] - b_center[1]) ** 2)
        class_id = b_center[2]  # 假设我们在 b_centers 中存储了 class_id
        hero_name = class_names.get(class_id, "未知英雄")

        if hero_name in priority_heroes:
            priority_targets.append((b_center, distance))
        elif not priority_targets and (closest_target is None or distance < min_distance):
            closest_target = b_center
            min_distance = distance

    if priority_targets:
        # 如果有先目标，选择最近的优先目标
        target, _ = min(priority_targets, key=lambda x: x[1])
    else:
        target = closest_target

    if target:
        g_grid = convert_to_grid_coordinates(g_center[0], g_center[1])
        b_grid = convert_to_grid_coordinates(target[0], target[1])
        path = a_star(g_grid, b_grid, obstacle_map)
        if path:
            print(f"[{time.time()}] 找到路径，跟随目标: {target}")
            return path

    print(f"[{time.time()}] 未找到可用路径")
    return None


def move_along_path_thread():
    """在线程中处理沿路径移动"""
    global g_center_cache, g_center_last_update_time
    stuck_threshold_time = 0.8  # 减少卡住判定时间
    stuck_threshold_distance = 8  # 略微减少卡住判定距离
    recalculation_interval = 0.4  # 减少重新计算间隔
    last_recalculation_time = time.time()
    g_center_miss_count = 0  # 添加计数器

    def is_stuck(elapsed_time, initial_distance, current_distance):
        return (elapsed_time > stuck_threshold_time and
                (initial_distance - current_distance) < stuck_threshold_distance)

    while not terminate_event.is_set():
        update_event.wait()
        update_event.clear()

        if not latest_path:
            continue

        current_step_index = 1
        stuck_start_time = time.time()

        while current_step_index < len(latest_path):
            next_step = latest_path[current_step_index]
            world_x, world_y = next_step[0] * CELL_SIZE + CELL_SIZE // 2, next_step[1] * CELL_SIZE + CELL_SIZE // 2
            initial_distance = None

            while True:
                with g_center_lock:
                    local_g_center = g_center

                if local_g_center is None:
                    g_center_miss_count += 1
                    current_time = time.time()
                    if g_center_cache and (current_time - g_center_last_update_time) < G_CENTER_CACHE_DURATION:
                        local_g_center = g_center_cache
                        print(f"[{current_time}] 使用缓存的 g_center 位置: {local_g_center}")
                    elif g_center_miss_count >= G_CENTER_MISS_THRESHOLD:
                        print(f"[{current_time}] g_center 连续 {G_CENTER_MISS_THRESHOLD} 次未检测到，停止跟随路径，当前索引 {current_step_index}")
                        check_and_clear_path()
                        break
                    else:
                        print(f"[{current_time}] g_center 暂时未检测到，继续使用上一个有效位置")
                        continue
                else:
                    g_center_miss_count = 0  # 重置计数器
                    g_center_cache = local_g_center
                    g_center_last_update_time = time.time()
                    print(f"[{time.time()}] 检测到 g_center 位置: {local_g_center}")

                dx, dy = world_x - local_g_center[0], world_y - local_g_center[1]
                current_distance = calculate_distance(local_g_center[0], local_g_center[1], world_x, world_y)

                if initial_distance is None:
                    initial_distance = current_distance

                if current_distance < 5:
                    current_step_index += 1
                    stuck_start_time = time.time()
                    print(f"[{time.time()}] 移动到下一个路径点，当前索引: {current_step_index}")
                    break

                elapsed_time = time.time() - stuck_start_time
                if is_stuck(elapsed_time, initial_distance, current_distance) and \
                        (time.time() - last_recalculation_time) > recalculation_interval:
                    with g_center_lock:
                        if g_center is not None:
                            g_grid = convert_to_grid_coordinates(g_center[0], g_center[1])
                            goal_grid = convert_to_grid_coordinates(world_x, world_y)
                            print(f"[{time.time()}] 重新计算路径，g_center 位置: {g_center}, 当前目标: ({world_x}, {world_y})")
                    new_path = a_star(g_grid, goal_grid, obstacle_map)
                    if new_path:
                        print(f"[{time.time()}] 发现新路径，开始沿新路径移动")
                        start_move_along_path(new_path)
                        last_recalculation_time = time.time()
                    else:
                        print(f"[{time.time()}] 路径重新计算失败")
                    break

                move_direction(dx, dy)
                time.sleep(0.008)  # 略微减少睡眠时间，提高响应速度

    print("路径跟随线程结束")

# 主循环
try:
    move_thread = threading.Thread(target=move_along_path_thread)
    move_thread.start()

    last_g_detected_time = time.time()

    window_name = "YOLOv8 Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1930, 250)
    cv2.resizeWindow(window_name, 350, 350)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    g_center_miss_count = 0  # 添加计数器

    while True:
        start_time = time.time()

        # 捕获屏幕指定区域并运行YOLO检测
        sct_img = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        results = model.track(frame, persist=True)

        g_center = None
        b_centers = []

        # 创建一个新的帧来绘制结果
        annotated_frame = frame.copy()

        try:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()

                for i, box in enumerate(boxes):
                    class_id = class_ids[i]
                    class_name = class_names.get(class_id, f"未知类别 {class_id}")
                    confidence = confidences[i]
                    x, y, w, h = box

                    # 获取框的颜色
                    box_color = get_color(class_id)

                    # 绘制框
                    cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), box_color, 2)

                    # 添加白色中文标签
                    label = f"{class_name} {confidence:.2f}"
                    annotated_frame = cv2_add_chinese_text(annotated_frame, label, (int(x-w/2), int(y-h/2-30)))

                    if 0 <= class_id <= 122 and confidence > confidence_threshold:
                        g_center = (x, y)
                        last_g_detected_time = time.time()
                    elif 123 <= class_id <= 245:
                        b_centers.append((x, y, class_id))  # 存储 class_id

            # 路径更新逻辑
            if g_center and b_centers:
                path = find_priority_target(b_centers, g_center)
                if path:
                    start_move_along_path(path)
                else:
                    check_and_clear_path()  # 如果路径不可用，清空路径
            else:
                check_and_clear_path()  # 如果检测不到目标，清空路径并停止移动

            if g_center is None:
                g_center_miss_count += 1
                current_time = time.time()
                if g_center_cache and (current_time - g_center_last_update_time) < G_CENTER_CACHE_DURATION:
                    g_center = g_center_cache
                    print(f"[{current_time}] 使用缓存的 g_center 位置: {g_center}")
                elif g_center_miss_count >= G_CENTER_MISS_THRESHOLD:
                    print(f"[{current_time}] g_center 连续 {G_CENTER_MISS_THRESHOLD} 次未检测到")
                    check_and_clear_path()
                else:
                    print(f"[{current_time}] g_center 暂时未检测到，继续使用上一个有效位置")
            else:
                g_center_miss_count = 0  # 重置计数器
                g_center_cache = g_center
                g_center_last_update_time = time.time()

        except Exception as e:
            print(f"处理检测结果时出错: {e}")

        # 绘制路径
        if latest_path:
            draw_path_on_image(annotated_frame, latest_path)

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed_time = time.time() - start_time
        time.sleep(max(0, frame_time - elapsed_time))

except KeyboardInterrupt:
    pass
finally:
    terminate_event.set()
    move_thread.join()
    cv2.destroyAllWindows()
