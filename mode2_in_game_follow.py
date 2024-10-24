import pyautogui
from math import sqrt

# 初始化 pyautoguiaaas
pyautogui.PAUSE = 0

# 定义常量
FOLLOW_DISTANCE = 20  # 跟随距离阈值

# 全局变量
last_movement = {'w': False, 's': False, 'a': False, 'd': False}

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

def find_closest_target(self_pos, targets):
    if not targets:
        return None
    return min(targets, key=lambda pos: sqrt((self_pos[0] - pos[0])**2 + (self_pos[1] - pos[1])**2))

def model2_movement_logic(detection_result):
    self_pos = detection_result['self_pos']
    team_targets = detection_result['team_targets']

    if self_pos:
        if team_targets:
            target = find_closest_target(self_pos, team_targets)
            if target:
                dx, dy = target[0] - self_pos[0], target[1] - self_pos[1]
                distance = sqrt(dx**2 + dy**2)
                if distance > FOLLOW_DISTANCE:
                    move_direction(dx, dy)
                else:
                    release_all_keys()
            else:
                release_all_keys()
        else:
            release_all_keys()
    else:
        release_all_keys()

    return {
        'self_pos': self_pos,
        'closest_target': target if 'target' in locals() else None,
        'is_moving': any(last_movement.values())
    }
