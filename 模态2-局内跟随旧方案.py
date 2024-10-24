from ultralytics import YOLO
import cv2
import numpy as np
import mss
import time
from math import sqrt
import pyautogui

# 初始化设置
pyautogui.PAUSE = 0
model = YOLO(r'./WZRY-health.pt')

# 设置捕获屏幕区域
sct = mss.mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# 定义常量
FPS = 30
FRAME_TIME = 1 / FPS
CONFIDENCE_THRESHOLD = 0.70
FOLLOW_DISTANCE = 200  # 跟随距离阈值

# 定义类别名称
class_names = ["g_self_health_health", "b_team_health", "b_low_health", "g_in_head_health", "g_in_head_low_health", "r_enemy_health"]

# 全局变量
last_movement = {'w': False, 's': False, 'a': False, 'd': False}
g_center_cache = None
g_center_last_update_time = 0
G_CENTER_TOLERANCE = 0.1  # 100毫秒的容忍期

def detect_objects(frame):
    results = model.track(frame, persist=True)
    return process_results(results)

def process_results(results):
    objects = {
        'g_self': [],
        'b_team': [],
        'r_enemy': []
    }

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            if confidence > CONFIDENCE_THRESHOLD:
                class_name = class_names[class_id]
                x, y, w, h = box
                if 'g_self' in class_name:
                    objects['g_self'].append((x, y))
                elif 'b_team' in class_name or 'b_low' in class_name:
                    objects['b_team'].append((x, y))
                elif 'r_enemy' in class_name:
                    objects['r_enemy'].append((x, y))

    return objects

def find_closest_target(self_pos, targets):
    if not targets:
        return None
    return min(targets, key=lambda pos: sqrt((self_pos[0] - pos[0])**2 + (self_pos[1] - pos[1])**2))

def move_direction(dx, dy):
    global last_movement
    current_keys = {'w': False, 's': False, 'a': False, 'd': False}
    abs_dx, abs_dy = abs(dx), abs(dy)

    if abs_dx > 5 and abs_dy > 5:
        current_keys['d'] = dx > 0
        current_keys['a'] = dx < 0
        current_keys['s'] = dy > 0
        current_keys['w'] = dy < 0
    elif abs_dx > abs_dy:
        current_keys['d'] = dx > 0
        current_keys['a'] = dx < 0
    else:
        current_keys['s'] = dy > 0
        current_keys['w'] = dy < 0

    for key in current_keys:
        if current_keys[key] != last_movement[key]:
            if current_keys[key]:
                pyautogui.keyDown(key)
            else:
                pyautogui.keyUp(key)
            last_movement[key] = current_keys[key]

def release_all_keys():
    global last_movement
    for key in last_movement:
        if last_movement[key]:
            pyautogui.keyUp(key)
            last_movement[key] = False

def main():
    global g_center_cache, g_center_last_update_time

    try:
        while True:
            start_time = time.time()

            # 捕获屏幕
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # 检测对象
            objects = detect_objects(frame)

            # 处理检测结果
            if objects['g_self']:
                self_pos = objects['g_self'][0]
                g_center_cache = self_pos
                g_center_last_update_time = time.time()
                
                if objects['b_team']:
                    target = find_closest_target(self_pos, objects['b_team'])
                    if target:
                        dx, dy = target[0] - self_pos[0], target[1] - self_pos[1]
                        distance = sqrt(dx**2 + dy**2)
                        if distance > FOLLOW_DISTANCE:
                            move_direction(dx, dy)
                        else:
                            release_all_keys()
                else:
                    release_all_keys()
            elif g_center_cache and time.time() - g_center_last_update_time < G_CENTER_TOLERANCE:
                self_pos = g_center_cache
            else:
                release_all_keys()

            # 显示结果（可选）
            annotated_frame = model.track(frame, persist=True)[0].plot()
            cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_frame, (960, 540)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 控制帧率
            time.sleep(max(0, FRAME_TIME - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        release_all_keys()
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    main()
