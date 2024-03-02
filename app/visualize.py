import csv
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import model.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.models import *
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable


def pred_faceExp(pic_path, result_path, pic_name):
    cut_size = 44

    transform_test = transforms.Compose(
        [
            transforms.TenCrop(cut_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
        ]
    )

    # 轉灰階
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    raw_img = io.imread(os.path.join(pic_path, pic_name))
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode="symmetric").astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    net = VGG("VGG19")
    checkpoint = torch.load(
        os.path.join("app", "model", "PrivateTest_model.t7"),
        map_location=torch.device("cpu"),  # cpu/cuda
    )
    net.load_state_dict(checkpoint["net"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.to(device)
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg, dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)

    # results img
    plt.rcParams["figure.figsize"] = (13.5, 5.5)

    # 第一張子圖
    axes = plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel("Input Image", fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3
    )

    # 第二張子圖
    plt.subplot(1, 3, 2)
    ind = 0.1 + 0.6 * np.arange(len(class_names))  # the x locations for the groups
    width = 0.4  # the width of the bars: can also be len(x) sequence
    color_list = [
        "red",
        "orangered",
        "darkorange",
        "limegreen",
        "darkgreen",
        "royalblue",
        "navy",
    ]
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ", fontsize=20)
    plt.xlabel(" Expression Category ", fontsize=16)
    plt.ylabel(" Classification Score ", fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    # 第三張子圖
    axes = plt.subplot(1, 3, 3)
    emojis_img = io.imread(
        "app/static/img/emojis/%s.png" % str(class_names[int(predicted.cpu().numpy())])
    )
    plt.imshow(emojis_img)
    plt.xlabel("Emoji Expression", fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    # show emojis

    plt.savefig(os.path.join(result_path, os.path.basename(pic_name)))
    plt.close()

    # csv
    csv_file = "app/data/emotion_ratios.csv"  # CSV文件的文件名
    column_names = ["Image"] + class_names  # 列名設置為情緒列表

    emotion_ratios = [float(score[i]) for i in range(len(score))]

    if not os.path.exists(csv_file):
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([os.path.basename(pic_name)] + emotion_ratios)

    predicted_emotion = class_names[int(predicted.cpu().numpy())]
    print("The Expression is %s" % predicted_emotion)
