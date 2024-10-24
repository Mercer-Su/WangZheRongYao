import threading
import time
from queue import Queue
import mss

print("开始导入模块")
import model1_astar_follow
import mode2_in_game_follow
print("所有模块导入完成")

def model1_thread(queue):
    print("模态1线程准备开始运行")
    with mss.mss() as sct:
        try:
            model1_astar_follow.run(queue, sct)
        except Exception as e:
            print(f"模态1线程发生异常: {e}")
    print("模态1线程结束")

def model2_thread(queue):
    print("模态2线程准备开始运行")
    with mss.mss() as sct:
        try:
            mode2_in_game_follow.run(queue, sct)
        except Exception as e:
            print(f"模态2线程发生异常: {e}")
    print("模态2线程结束")

def main():
    print("进入main函数")
    model1_queue = Queue()
    model2_queue = Queue()

    print("创建线程")
    thread1 = threading.Thread(target=model1_thread, args=(model1_queue,), daemon=True)
    thread2 = threading.Thread(target=model2_thread, args=(model2_queue,), daemon=True)

    print("准备启动模态1线程")
    thread1.start()
    print("模态1线程已启动")

    print("准备启动模态2线程")
    thread2.start()
    print("模态2线程已启动")

    current_mode = 2  # 初始模式设为模态2

    try:
        print("进入主循环")
        while True:
            if not model2_queue.empty():
                model2_result = model2_queue.get()
                print("模态2消息:", model2_result)
                if model2_result['self_pos'] and not model2_result['team_targets']:
                    if current_mode != 1:
                        print("切换到模态1")
                        current_mode = 1
                        model1_queue.put({'activate': True})
                elif model2_result['team_targets']:
                    if current_mode != 2:
                        print("切换到模态2")
                        current_mode = 2
                        model1_queue.put({'activate': False})

            if current_mode == 1 and not model1_queue.empty():
                model1_result = model1_queue.get()
                print("模态1消息:", model1_result)

            time.sleep(0.01)  # 短暂休眠以减少CPU使用
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("程序结束")

if __name__ == "__main__":
    print("程序开始")
    main()
