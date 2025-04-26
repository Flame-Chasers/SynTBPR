import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, CLIPModel

# 加载模型和处理器
model_path = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# 定义图像文件夹路径
image_folder_path = "../datasets/CUHK-PEDES/imgs"

# 获取文件夹中所有图片的路径
image_paths = [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith(('png', 'jpg', 'jpeg'))]

# 提取每张图片的特征
image_features_list = []

for image_path in image_paths:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).numpy().flatten()
    image_features_list.append(image_features)

# 将特征列表转换为numpy数组
image_features_matrix = np.array(image_features_list)

# 计算特征的方差
variance_vector = np.var(image_features_matrix)

print("每个特征的方差: ", variance_vector)
