import os
import re
import sys
import threading
import warnings
from datetime import datetime

# 取得 visualize.py 所在的目錄路徑
module_path = os.path.dirname(os.path.abspath(__file__))
# 將模組所在的目錄路徑加入 Python 的搜尋路徑
sys.path.append(module_path)

import multiprocessing

import genderAge as modelG
import matplotlib
import video_processing
import visualize as model
from flask import Flask, render_template, request, session

warnings.filterwarnings("ignore")

matplotlib.use("Agg")

UPLOAD_FOLDER = "app/static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
cutting_path = "app/static/cutting_img/"
grep_face_path = "app/static/grep_face/"
result_path = "app/static/results_img/"


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)

app.secret_key = "FacialExpressionRecognition"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/inner-page", methods=["GET", "POST"])
def upload():
    # 進入 upload() 時清空 session
    session.clear()

    if request.method == "GET":
        return render_template("inner-page.html")

    elif request.method == "POST":
        file = request.files["video"]
        if file:
            # 使用當前時間來生成唯一的檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            session["timestamp"] = timestamp  # 存入 session

            # 儲存錄影檔
            filename = f"{timestamp}.webm"
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            # Way1: 轉編碼 -> opencv cutting
            # # 錄影檔編碼格式轉換
            # input_file = os.path.join(UPLOAD_FOLDER, filename)
            # output_file = os.path.join("app/static/output_video/", filename)
            # video_processing.convert_video_to_h264(input_file, output_file, 30)

            # # Cutting
            # video_processing.process_video(output_file, cutting_path)


            # Way2: ffmpeg cutting -> opencv grep face
            # Cutting
            video_processing.process_video(
                os.path.join(UPLOAD_FOLDER, filename), cutting_path
            )

            # grep face
            for n in range(5):
                # 假設每個 segment 最多有 5 張照片
                for m in range(5):
                    rec_filename = f"{timestamp}_segment_{n+1}_frame_{m+1}.jpg"  # 圖片名稱
                    try:
                        video_processing.grep_face(
                            cutting_path, grep_face_path, rec_filename
                        )
                    except FileNotFoundError:
                        continue

            # 啟動多進程來處理 Facial-Expression-Recognition Model 的預測
            prediction_process = multiprocessing.Process(
                target=prediction, args=(timestamp,)
            )
            prediction_process.start()

            # 等待 prediction_process 執行完畢再返回模板
            prediction_process.join()

        return render_template("recog-result.html", timestamp=session["timestamp"])

    else:
        return "檔案上傳失敗"


# Facial-Expression-Recognition Model
def prediction(timestamp):
    for n in range(5):
        # 假設每個 segment 最多有 5 張照片
        for m in range(5):
            rec_filename = f"{timestamp}_segment_{n+1}_frame_{m+1}.jpg"  # 圖片名稱
            try:
                model.pred_faceExp(grep_face_path, result_path, rec_filename)
                break  # 找到第一張符合條件的照片後，跳出 m 迴圈
            except FileNotFoundError:
                continue


@app.route("/recog-result")
def result():
    timestamp = session.get("timestamp")  # 從 session 中取得 timestamp 變數
    print(timestamp)

    pattern = re.compile(rf"{timestamp}_segment_(\d+)_frame_(\d+)\.jpg")

    # 建立包含五個空列表的segment_files字典，用於儲存匹配的檔案名稱
    segment_files = {f"segment_{i}": [] for i in range(1, 6)}

    # 逐一處理每個matching_file
    for filename in os.listdir(result_path):
        if pattern.match(filename):
            match = pattern.match(filename)
            segment_num = int(match.group(1))

            # 將matching_file加入對應的陣列
            if 1 <= segment_num <= 5:
                segment_files[f"segment_{segment_num}"].append(filename)

    matching_files = [segment_files[f"segment_{i}"] for i in range(1, 6)]

    # 創建一個新的線程來執行 Gender-Age 的預測 (網頁顯示完再進 Gender-Age Model)
    prediction_thread = threading.Thread(target=genderAge_pred, args=(timestamp,))
    prediction_thread.start()

    return render_template("recog-result.html", report=matching_files)


# Gender-Age Model
def genderAge_pred(timestamp):
    for n in range(5):
        for m in range(33, 37):
            rec_filename = f"{timestamp}_segment_{n+1}_frame_{m}.jpg"
            try:
                modelG.genderAge(cutting_path, rec_filename)
                break  # 找到第一張符合條件的照片後，跳出 m 迴圈
            except FileNotFoundError:
                continue
