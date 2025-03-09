import threading
import tkinter as tk
import tkinter.messagebox as messagebox
import cv2
import os
import tkinter.filedialog as fd
from ultralytics import YOLO

running = True  # 控制預測狀態
smoke_count = 0  # 煙霧偵測次數
fire_count = 0   # 火焰偵測次數
stats_window = None  # 統計視窗
alert_window = None  # 警告視窗
alert_text = None  # 警告標語

# 🔹 定義類別顏色
class_colors = {
    0: (0, 255, 0),   # smoke → 綠色
    1: (0, 0, 255),   # fire → 紅色
    2: (255, 0, 0),   # 其他類別 → 藍色
}

def update_alert_window():
    """即時更新警告視窗內容"""
    global alert_window, alert_text

    if alert_window is not None:
        alert_msg = ""
        if fire_count > 0:
            alert_msg += "🔥 火焰警告！請立即檢查！🔥\n"
        if smoke_count > 0:
            alert_msg += "💨 煙霧警告！請立即檢查！💨"

        if alert_msg:
            alert_text.set(alert_msg)
        else:
            alert_text.set("✅ 目前無危險")

    # 每秒更新一次警告視窗
    if alert_window is not None:
        alert_window.after(1000, update_alert_window)  # 每1000毫秒（1秒）更新一次

def open_alert_window():
    """開啟警告視窗"""
    global alert_window, alert_text

    if alert_window is None or not alert_window.winfo_exists():
        alert_window = tk.Toplevel(root)
        alert_window.title("警告通知")
        alert_window.geometry("300x150")
        alert_window.configure(bg="black")

        alert_text = tk.StringVar()
        alert_label = tk.Label(
            alert_window, textvariable=alert_text, font=("Arial", 14, "bold"), 
            fg="red", bg="black", wraplength=280
        )
        alert_label.pack(pady=20)

        update_alert_window()  # 初始更新警告視窗

def update_stats_window():
    """更新統計視窗內容"""
    global stats_window
    if stats_window is not None:
        stats_text.set(f"🔥 火焰次數: {fire_count}\n💨 煙霧次數: {smoke_count}")

def open_stats_window():
    """開啟統計視窗"""
    global stats_window, stats_text

    if stats_window is None or not stats_window.winfo_exists():
        stats_window = tk.Toplevel(root)
        stats_window.title("偵測統計")
        stats_window.geometry("250x150")
        stats_text = tk.StringVar()
        stats_label = tk.Label(stats_window, textvariable=stats_text, font=("Arial", 14))
        stats_label.pack(pady=20)
        update_stats_window()

def run_prediction():
    """執行靜態圖片的 YOLO 預測"""
    global running, smoke_count, fire_count
    model = YOLO('Completeweight\8sbest100.pt') #在此更換訓練結果權重
    results = model.predict(source='images.jpg', imgsz=320, save=True, conf=0.5)
    
    # 取得 YOLO 存放結果的目錄
    output_dir = results[0].save_dir  
    output_path = os.path.join(output_dir, "images.jpg")  

    # 確保圖片存在
    if os.path.exists(output_path):
        img = cv2.imread(output_path)
        if img is not None:
            img = cv2.resize(img, (640, 640))  # 調整大小

            # 檢查是否偵測到 fire 或 smoke
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])  # 取得類別索引
                    if cls == 0:  # smoke
                        smoke_count += 1
                    elif cls == 1:  # fire
                        fire_count += 1

            open_alert_window()  # 彈出警告視窗，並更新次數

            cv2.imshow("YOLO Prediction", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("❌ 無法找到預測結果圖片:", output_path)

    running = True

def run_video_prediction():
    """開啟檔案對話框，選擇影片並進行 YOLO 偵測"""
    global smoke_count, fire_count
    file_path = fd.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    
    if not file_path:
        print("⚠️ 沒有選擇影片")
        return

    print(f"🎥 選擇的影片: {file_path}")

    model = YOLO('Completeweight\8sbest100.pt') #在此更換訓練結果權重

    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        print("❌ 無法開啟影片")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # 取得影片的 FPS
    delay = int(1000 / fps)  # 計算每一幀應該等待的時間 (毫秒)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # 直接對影片的每一幀執行 YOLO 預測
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得邊界框座標
                conf = box.conf[0]  # 置信度
                cls = int(box.cls[0])  # 類別索引
                class_names = model.names  # 取得類別名稱字典
                label = f"{class_names[cls]}: {conf:.2f}"  # 轉換成真實名稱
                
                # 🔹 根據類別選擇不同顏色，若類別不存在則預設為白色
                color = class_colors.get(cls, (255, 255, 255))  

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 畫框
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 偵測到火焰或煙霧時，增加次數
                if cls == 0:
                    smoke_count += 1
                elif cls == 1:
                    fire_count += 1

        open_alert_window()  # 更新警告視窗

        cv2.imshow("YOLO Video Prediction", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_prediction():
    """手動中止 YOLO 預測"""
    global running
    running = False
    print("\n🔴 預測已手動中止！")
    root.quit()  

# 🔹 建立 GUI 視窗
root = tk.Tk()
root.title("YOLO 物件偵測")
root.geometry("400x400")  # 設定視窗大小

# 🔹 設定按鈕大小、字體、間距
start_button = tk.Button(
    root, text="偵測圖片", command=lambda: threading.Thread(target=run_prediction).start(),
    width=20, height=2, font=("Arial", 12), padx=10, pady=5
)
start_button.pack(pady=15)  

video_button = tk.Button(
    root, text="偵測影片", command=lambda: threading.Thread(target=run_video_prediction).start(),
    width=20, height=2, font=("Arial", 12), padx=10, pady=5
)
video_button.pack(pady=15)  

stats_button = tk.Button(
    root, text="查看偵測次數", command=open_stats_window,
    width=20, height=2, font=("Arial", 12), padx=10, pady=5
)
stats_button.pack(pady=15)

stop_button = tk.Button(
    root, text="停止偵測", command=stop_prediction,
    width=20, height=2, font=("Arial", 12), padx=10, pady=5
)
stop_button.pack(pady=15)  

root.mainloop()
