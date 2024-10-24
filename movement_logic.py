import threading
import time
from queue import Queue, Empty
import model1_detector
import model2_detector
import model1_astar_follow
import mode2_in_game_follow
import mss
import cv2
from math import sqrt
import yao_skill_logic

def model1_thread(queue, result_queue):
    print("模态1线程开始运行")
    with mss.mss() as sct:
        while True:
            detection_result = model1_detector.detect(sct)
            movement_result = model1_astar_follow.model1_movement_logic(detection_result)
            result_queue.put(('model1', detection_result, movement_result))
            print(f"模态1发送检测结果: g_center={detection_result['g_center']}, b_centers数量={len(detection_result['b_centers'])}")
            time.sleep(0.01)

def model2_thread(queue, result_queue):
    print("模态2线程开始运行")
    with mss.mss() as sct:
        while True:
            detection_result = model2_detector.detect(sct)
            movement_result = mode2_in_game_follow.model2_movement_logic(detection_result)
            result_queue.put(('model2', detection_result, movement_result))
            time.sleep(0.01)

def find_closest_target(self_pos, targets):
    if not targets:
        return None
    return min(targets, key=lambda pos: sqrt((self_pos[0] - pos[0])**2 + (self_pos[1] - pos[1])**2))

def main():
    model1_queue = Queue()
    model2_queue = Queue()
    result_queue = Queue()
    yao_skill_queue = Queue(maxsize=1)  # 限制队列大小为1

    thread1 = threading.Thread(target=model1_thread, args=(model1_queue, result_queue), daemon=True)
    thread2 = threading.Thread(target=model2_thread, args=(model2_queue, result_queue), daemon=True)
    yao_thread = threading.Thread(target=yao_skill_logic.run, args=(yao_skill_queue,), daemon=True)

    thread1.start()
    thread2.start()
    yao_thread.start()

    current_mode = 2  # 初始模式设为模态2
    last_check_time = time.time()
    force_mode1_until = 0
    last_health_info = None

    try:
        while True:
            model1_result = None
            model2_result = None

            try:
                while not result_queue.empty():
                    model, detection, movement = result_queue.get(timeout=0.1)
                    if model == 'model1':
                        model1_result = (detection, movement)
                    elif model == 'model2':
                        model2_result = (detection, movement)
            except Empty:
                pass

            current_time = time.time()

            # 高优先级切换逻辑
            if current_time - last_check_time >= 10:
                last_check_time = current_time
                if model2_result and not model2_result[0]['enemies']:
                    if model1_result:
                        priority_hero_detected = any(model1_astar_follow.class_names.get(b[2], "") in model1_astar_follow.priority_heroes for b in model1_result[0]['b_centers'])
                        if priority_hero_detected:
                            current_mode = 1
                            force_mode1_until = current_time + 10
                            print("检测到优先英雄，强制切换到模态1")

            # 常规模式切换逻辑
            if current_time >= force_mode1_until:
                if model2_result and model2_result[0]['team_targets']:
                    if current_mode != 2:
                        print("检测到队友血条，切换到模态2")
                        current_mode = 2
                        model1_queue.put({'activate': False})
                elif current_mode == 2 and (not model2_result or not model2_result[0]['team_targets']):
                    print("未检测到队友血条，切换回模态1")
                    current_mode = 1
                    model1_queue.put({'activate': True})

            # 执行当前模式的移动逻辑
            if current_mode == 1 and model1_result:
                print("使用模态1移动逻辑")
                detection, movement = model1_result
                if movement['is_moving']:
                    print(f"模态1正在移动，目标: {movement['closest_b']}")
                else:
                    print("模态1未移动")
                
                # 在模态1下发送空的血量信息给瑶的技能逻辑
                health_info = {
                    'self_health': None,
                    'team_health': [],
                    'enemy_health': [],
                    'enemy_positions': [],
                    'team_positions': []
                }
                
                # 只有当信息发生变化时才发送
                if health_info != last_health_info:
                    # 清空队列并添加新的信息
                    while not yao_skill_queue.empty():
                        try:
                            yao_skill_queue.get_nowait()
                        except Empty:
                            pass
                    yao_skill_queue.put(health_info)
                    last_health_info = health_info

            elif current_mode == 2 and model2_result:
                print("使用模态2移动逻辑")
                detection, movement = model2_result
                self_pos = detection['self_pos']
                self_health = detection.get('self_health')
                team_targets = detection['team_targets']
                enemies = detection['enemies']
                
                # 构建健康信息
                health_info = {
                    'self_health': self_health,
                    'team_health': [target[2] for target in team_targets],
                    'enemy_health': [enemy[2] for enemy in enemies],
                }

                # 只有当 self_pos 不为 None 时才计算距离
                if self_pos is not None:
                    health_info['enemy_positions'] = [sqrt((self_pos[0] - e[0])**2 + (self_pos[1] - e[1])**2) for e in enemies]
                    health_info['team_positions'] = [sqrt((self_pos[0] - t[0])**2 + (self_pos[1] - t[1])**2) for t in team_targets]
                else:
                    health_info['enemy_positions'] = []
                    health_info['team_positions'] = []
                
                # 只有当信息发生变化时才发送
                if health_info != last_health_info:
                    # 清空队列并添加新的信息
                    while not yao_skill_queue.empty():
                        try:
                            yao_skill_queue.get_nowait()
                        except Empty:
                            pass
                    yao_skill_queue.put(health_info)
                    last_health_info = health_info

                # 输出模态2的详细检测结果
                print(f"模态2检测结果:")
                print(f"  自身位置: {self_pos}")
                print(f"  自身血量: {self_health if self_health is not None else 'None'}")
                print(f"  队友数量: {len(team_targets)}")
                for i, target in enumerate(team_targets):
                    print(f"    队友{i+1}: 位置={target[:2]}, 血量={target[2]}%")
                print(f"  敌人数量: {len(enemies)}")
                for i, enemy in enumerate(enemies):
                    print(f"    敌人{i+1}: 位置={enemy[:2]}, 血量={enemy[2]}%")

                # 添加检查，确保 self_pos 不为 None
                if self_pos is not None:
                    if team_targets:
                        closest_target = find_closest_target(self_pos, team_targets)
                        if closest_target:
                            dx, dy = closest_target[0] - self_pos[0], closest_target[1] - self_pos[1]
                            mode2_in_game_follow.move_direction(dx, dy)
                            print(f"跟随最近的队友，移动方向: dx={dx:.2f}, dy={dy:.2f}")
                        else:
                            mode2_in_game_follow.release_all_keys()
                            print("没有可跟随的队友")
                    else:
                        mode2_in_game_follow.release_all_keys()
                        print("未检测到队友")
                else:
                    mode2_in_game_follow.release_all_keys()
                    print("未检测到自身位置")

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()
