import pyautogui
import time
from queue import Queue, Empty

class YaoSkillLogic:
    def __init__(self):
        self.health_info = None

    def use_skill(self, skill):
        pyautogui.keyDown(skill)
        time.sleep(0.1)  # 短暂按下以确保游戏注册到按键
        pyautogui.keyUp(skill)
        print(f"使用技能 {skill}")
        if skill == 'r':
            time.sleep(0.5)  # r 技能后额外等待

    def check_and_use_skills(self):
        if not self.health_info:
            return

        self_health = self.health_info.get('self_health')
        team_health = self.health_info.get('team_health', [])
        enemy_health = self.health_info.get('enemy_health', [])
        enemy_positions = self.health_info.get('enemy_positions', [])
        team_positions = self.health_info.get('team_positions', [])

        print(f"当前健康信息: 自身血量={self_health}, 队友数量={len(team_health)}, 敌人数量={len(enemy_health)}")

        # 只有在检测到自身位置时才执行技能逻辑
        if self_health is not None:
            print("检测到自身位置，执行技能逻辑：", self_health)
            # 1. 逃跑逻辑
            if enemy_positions and not team_positions:
                pyautogui.keyDown('a')
                pyautogui.keyDown('s')
                pyautogui.keyUp('w')
                pyautogui.keyUp('d')
                print("润!")
                return

            # 2. 辅助装备和治疗使用
            low_health_allies = [health for health in team_health if health < 70]
            if low_health_allies:
                self.use_skill('p')
                print("辅助装回血")
                if enemy_positions and min(low_health_allies) < 30:
                    self.use_skill('f')
                    print("治疗")

            if team_positions and min(team_positions) < 2500:
                self.use_skill('r')
                self.use_skill('e')
                self.use_skill('q')
                print("上身r，释放1，2技能")
        else:
            self.use_skill('e')
            self.use_skill('q')
            self.use_skill('c')
            print("释放1,2,恢复技能")
            print("未检测到自身位置，视为附身状态")

    def run(self, queue):
        while True:
            try:
                self.health_info = queue.get(timeout=0.1)
                print("瑶技能逻辑接收到的信息:", self.health_info)
                self.check_and_use_skills()
                self.health_info = None  # 清空当前状态
            except Empty:
                pass
            time.sleep(0.1)  # 每0.1秒检查一次

def run(queue):
    yao = YaoSkillLogic()
    yao.run(queue)

if __name__ == "__main__":
    test_queue = Queue()
    run(test_queue)
