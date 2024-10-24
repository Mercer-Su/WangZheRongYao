import pygetwindow as gw
import time

def move_window_to_corner(window_title):
    try:
        # 尝试获取指定标题的窗口
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            window = windows[0]
            # 将窗口移动到左上角 (0, 0) 位置
            window.moveTo(0, 0)
            print(f"已将窗口 '{window_title}' 移动到左上角。")
        else:
            print(f"未找到标题为 '{window_title}' 的窗口。")
    except Exception as e:
        print(f"移动窗口时发生错误: {e}")

if __name__ == "__main__":
    # 等待几秒，以便观察结果
    time.sleep(3)
    window_title = "雷电模拟器"  # 这里设置为你的目标窗口标题
    move_window_to_corner(window_title)
    
    # 等待几秒，以便观察结果
    time.sleep(3)
