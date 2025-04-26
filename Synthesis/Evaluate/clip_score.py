import json
import os
import time
import glob
import torch
from PIL import Image
import cv2
import numpy as np
import random
from torchmetrics.multimodal.clip_score import CLIPScore
import torchvision.transforms as transforms
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric = CLIPScore(model_name_or_path="clip-vit-large-patch14").to(device)
all_json = ["path.json"]
all_json = sorted(all_json)
for json_path in all_json:
    json_name = json_path.split('/')[-1]
    with open(json_path, 'r') as f:
        data = json.load(f)
    scores = []
    begin = time.time()
    idx = 0
        # 用于存储每个batch的图像和文本
    batch_images = []
    batch_texts = []
    batch_size = 64
    for ann in data:
        image_path = ann['image_path']
        text = ann['caption']

        opencv_image = cv2.imread(image_path)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        opencv_image = torch.from_numpy(opencv_image).permute(2, 0, 1).to(device)  # 单个图像转换为张量
        batch_images.append(opencv_image)
        batch_texts.append(text)
        
        if len(batch_images) == batch_size:
            # 处理当前批次
            images = batch_images  # 批量组合
            texts = batch_texts  # 假设 metric 可以处理批量文本

            # 计算分数
            with torch.no_grad():
                scores_batch = metric(images, texts).cpu().detach().numpy()
                # print(scores_batch)
            scores.append(scores_batch)
            batch_images = []
            batch_texts = []

            idx += batch_size
            if idx % 500 == 0:
                print("iter = {} time is {}".format(idx, time.time() - begin))

    # 处理剩余的图像
    if len(batch_images) > 0:
        images = batch_images
        texts = batch_texts

        with torch.no_grad():
            scores_batch = metric(images, texts).cpu().detach().numpy()
        
        scores.append(scores_batch)
    if len(scores) > 0:
        print("{} is {}".format(json_name,sum(scores) / len(scores)))
    print("span time: {}".format(time.time() - begin))