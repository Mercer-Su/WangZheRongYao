import cv2
import numpy as np
import mss
from ultralytics import YOLO
import logging
import time

# 设置日志级别为 ERROR
logging.getLogger('ultralytics').setLevel(logging.ERROR)

scale = 0.35  # 窗口缩放比例
# 加载 YOLOv8 模型
model = YOLO(r'./WZRY-health.pt')  # 模型路径

# 定义常量
FPS = 30
FRAME_TIME = 1 / FPS
CONFIDENCE_THRESHOLD = 0.70

# 定义类别名称
class_names = ["g_self_health_health", "b_team_health", "b_low_health", "g_in_head_health", "g_in_head_low_health", "r_enemy_health"]

class HealthBar:
    def __init__(self, name, lower_hsv, upper_hsv, target_height, width_tolerance, height_tolerance, color, label):
        self.name = name
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.target_height = target_height
        self.width_tolerance = width_tolerance
        self.height_tolerance = height_tolerance
        self.color = color
        self.label = label

    def calculate_health_percentage(self, roi, detected_width):
        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, self.lower_hsv, self.upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            if (self.target_height - self.height_tolerance <= h <= self.target_height + self.height_tolerance):
                health_percentage = min(100, int((w / detected_width) * 100))
                if 95 <= health_percentage <= 100:
                    return 100
                return health_percentage

        return 100  # 如果未检测到符合条件的血条，返回100%

    def draw_health_bar(self, image, bbox, health_percentage, yolo_color, opencv_color):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), yolo_color, 2)
        text = f'{self.label} (YOLOv8)'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yolo_color, 2)
        filled_width = int((x2 - x1) * health_percentage / 100)
        cv2.rectangle(image, (x1, y1), (x1 + filled_width, y2), opencv_color, 2)
        text = f'Health: {health_percentage}%'
        cv2.putText(image, text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, opencv_color, 2)

def detect(sct):
    # 定义检测区域的坐标
    top_left = (0, 36)
    bottom_right = (1920, 1113)
    region = {
        "top": top_left[1],
        "left": top_left[0],
        "width": bottom_right[0] - top_left[0],
        "height": bottom_right[1] - top_left[1]
    }

    # 定义不同类型血条的 HSV 参数和颜色
    green_health_bar = HealthBar(
        name="Green Health Bar", lower_hsv=np.array([54, 154, 102]), upper_hsv=np.array([70, 255, 255]),
        target_height=10, width_tolerance=5, height_tolerance=5, color=(0, 255, 0), label='Self'
    )

    blue_health_bar = HealthBar(
        name="Blue Health Bar", lower_hsv=np.array([52, 76, 193]), upper_hsv=np.array([128, 201, 252]),
        target_height=13, width_tolerance=5, height_tolerance=4, color=(255, 0, 0), label='Team'
    )

    red_health_bar = HealthBar(
        name="Red Health Bar", lower_hsv=np.array([0, 40, 147]), upper_hsv=np.array([3, 255, 255]),
        target_height=13, width_tolerance=5, height_tolerance=4, color=(0, 0, 255), label='Enemy'
    )

    yolo_color = (255, 0, 0)  # 蓝色
    opencv_color = (0, 255, 0)  # 绿色

    # 获取屏幕截图
    screenshot = sct.grab(region)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 縮放圖像大小
    img_resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # 使用縮放後的圖像進行YOLO檢測
    img_for_yolo = cv2.resize(img, (640, 640))
    results = model.predict(img_for_yolo, conf=0.8, iou=0.5)

    # 将检测结果映射回原始图像尺寸
    scale_x = img.shape[1] / 640
    scale_y = img.shape[0] / 640

    self_pos = None
    team_targets = []
    enemies = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            class_id = int(box.cls[0])

            # 根据类别选择对应的 HealthBar 实例
            if class_id == 0:  # g_self_health
                health_bar = green_health_bar
                self_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
            elif class_id == 1:  # b_team_health
                health_bar = blue_health_bar
            elif class_id == 2:  # r_enemy_health
                health_bar = red_health_bar
            else:
                continue

            # 动态更新 target_width 为检测到的边界框宽度
            detected_width = x2 - x1

            # 截取检测到的 ROI 区域
            roi = img[y1:y2, x1:x2]

            # 计算血条的健康百分比
            health_percentage = health_bar.calculate_health_percentage(roi, detected_width)

            # 将血量信息添加到相应的列表中
            if class_id == 0:
                self_health = health_percentage
            elif class_id == 1:
                team_targets.append(((x1 + x2) / 2, (y1 + y2) / 2, health_percentage))
            elif class_id == 2:
                enemies.append(((x1 + x2) / 2, (y1 + y2) / 2, health_percentage))

            # 绘制 YOLOv8 检测框（蓝色）和 OpenCV 血量框（绿色）
            health_bar.draw_health_bar(img, (x1, y1, x2, y2), health_percentage, yolo_color, opencv_color)

    # 縮放顯示圖像
    img_resized_display = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # 显示图像
    cv2.imshow('Health Bar Detection', img_resized_display)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return None

    return {
        'self_pos': self_pos, 
        'self_health': self_health if 'self_health' in locals() else None,
        'team_targets': team_targets, 
        'enemies': enemies
    }

if __name__ == "__main__":
    with mss.mss() as sct:
        while True:
            result = detect(sct)
            if result is None:
                break
            print(result)
