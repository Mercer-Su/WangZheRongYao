import numpy as np
import logging
from ultralytics import YOLO
import cv2
import mss
import time
from PIL import Image, ImageDraw, ImageFont

# 设置日志级别为 ERROR
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# 加载YOLO模型
model = YOLO(r'best_perfect.pt')

# 捕获屏幕区域设置
monitor = {"top": 32, "left": 0, "width": 350, "height": 350}

# 帧率设置
fps = 30
frame_time = 1 / fps

# 加载中文字体
font_path = r"C:\Windows\Fonts\simhei.ttf"
font = ImageFont.truetype(font_path, 20)

# 加载真实类别名称
with open(r'name.txt', 'r', encoding='utf-8') as file:
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

# 置信度阈值
confidence_threshold = 0.90

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

def detect(sct):
    # 捕获屏幕指定区域并运行YOLO检测
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
    results = model.track(frame, persist=True)

    g_center = None
    b_centers = []
    r_centers = []

    # 创建一个新的帧来绘制结果
    annotated_frame = frame.copy()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            class_id = class_ids[i]
            confidence = confidences[i]
            x, y, w, h = box

            # 获取框的颜色
            box_color = get_color(class_id)

            # 绘制框
            cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), box_color, 2)

            # 添加白色中文标签
            class_name = class_names.get(class_id, f"未知类别 {class_id}")
            label = f"{class_name} {confidence:.2f}"
            annotated_frame = cv2_add_chinese_text(annotated_frame, label, (int(x-w/2), int(y-h/2-30)))

            if 0 <= class_id <= 122 and confidence > confidence_threshold:
                g_center = (x, y)
            elif 123 <= class_id <= 245:
                b_centers.append((x, y, class_id))  # 添加 class_id
            elif class_id > 245:
                r_centers.append((x, y, class_id))  # 添加 class_id

    # 显示图像
    cv2.imshow("Model 1 Detection", annotated_frame)
    cv2.waitKey(1)

    return {
        'g_center': g_center, 
        'b_centers': b_centers, 
        'r_centers': r_centers,
        'annotated_frame': annotated_frame,
        'class_names': class_names  # 添加 class_names 字典
    }

if __name__ == "__main__":
    with mss.mss() as sct:
        while True:
            result = detect(sct)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
