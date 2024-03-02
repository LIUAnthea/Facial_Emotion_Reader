import os
import subprocess

import cv2

# Way1: 轉編碼 -> opencv cutting
# # 錄影檔編碼格式轉換
# def convert_video_to_h264(input_file, output_file, output_frame_rate):
#     # 讀取影片檔案
#     cap = cv2.VideoCapture(input_file)

#     # 獲取影片的基本資訊
#     # frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # 使用 MPEG-4 編碼格式進行轉換
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output_file, fourcc, output_frame_rate, (width, height))

#     # 讀取並寫入每一禎影像
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)

#     # 釋放資源並關閉影片檔案
#     cap.release()
#     out.release()

#     return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 影片時長（秒）


# # Cutting
# def process_video(video_path, save_path):
#     cap = cv2.VideoCapture(video_path)

#     # 抓出影片檔名
#     video_filename = os.path.basename(video_path)
#     video_filename = os.path.splitext(video_filename)[0]  # 移除副檔名部分

#     # 確認影片成功開啟
#     if not cap.isOpened():
#         print("無法開啟影片檔案")
#         return
#     # cascade classifier
#     face_cascade = cv2.CascadeClassifier(
#         "app/model/haarcascade_frontalface_default.xml"
#     )
#     # 取得影片資訊
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 影片的幀率
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 影片的總幀數
#     duration = total_frames / frame_rate  # 影片的長度（秒）
#     print("影片幀率= ", frame_rate)
#     print("影片總幀數= ", total_frames)
#     print("影片長度= ", duration)

#     # 計算每段影片的開始和結束幀數
#     segment_length = 1.7  # 每段影片的長度（秒）
#     segment_frames = int(frame_rate * segment_length)  # 每段影片的幀數
#     # num_segments = int(duration / segment_length)  # 總段數

#     # 處理每一段影片
#     for i in range(5):
#         # 設定影片指標位置
#         start_frame = i * segment_frames + 5.5 * frame_rate
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#         # 處理每一禎影片
#         for j in range(segment_frames):
#             ret, frame = cap.read()
#             if ret:
#                 # 只保存每段影片的前 5 禎
#                 if j <= 35 & j >= 31:
#                     # 人臉偵測
#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     faces = face_cascade.detectMultiScale(
#                         gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
#                     )

#                     # 擷取人臉並保存圖片
#                     for x, y, w, h in faces:
#                         face_image = frame[y : y + h, x : x + w]
#                         # 生成儲存的檔案路徑
#                         output_path = os.path.join(
#                             save_path, f"{video_filename}_segment_{i+1}_frame_{j+1}.jpg"
#                         )
#                         # 將人臉圖片儲存為 jpg 檔案
#                         cv2.imwrite(output_path, face_image)
#             else:
#                 print(f"無法讀取第{i+1}段第{j+1}禎影片")

#     # 釋放資源並關閉影片檔案
#     cap.release()



# Way2: ffmpeg cutting -> opencv grep face
# Cutting
def process_video(video_path, save_path):
    # FFmpeg
    # 指定每段影片的開始秒數和間隔
    segment_length = 1.7
    start_seconds = [5.5 + i * segment_length for i in range(5)]

    # 使用ffmpeg進行影像擷取
    for i, start_time in enumerate(start_seconds):
        # 設定輸出圖片的檔案名稱
        output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_segment_{i+1}_frame_%d.jpg"
        output_path = os.path.join(save_path, output_filename)

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss",  # 起始時間
            str(start_time),
            "-i",  # 輸入影片檔案路徑
            video_path,
            "-vf",  # 選擇特定幀數的影格
            f"select='between(n,25,29)',setpts=N/FRAME_RATE/TB",
            "-q:v",  # 視訊品質
            "2",
            "-frames:v",  # 影格數量
            "5",
            output_path,  # 輸出圖片的檔案路徑
        ]

        subprocess.run(ffmpeg_cmd, capture_output=True)


# grep face
def grep_face(pic_path, save_path, pic_name):
    # 載入人臉偵測器
    face_cascade = cv2.CascadeClassifier(
        "app/model/haarcascade_frontalface_default.xml"
    )

    # 讀取照片
    img = cv2.imread(os.path.join(pic_path, pic_name))

    # 人脸偵測
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 擷取人臉並保存圖片
    for x, y, w, h in faces:
        # 儲存第一個偵測到的人臉
        x, y, w, h = faces[0]
        face_img = img[y : y + h, x : x + w]
        face_save_path = os.path.join(save_path, pic_name)
        cv2.imwrite(face_save_path, face_img)
