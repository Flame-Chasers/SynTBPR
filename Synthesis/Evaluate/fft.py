import os
import numpy as np
import cv2

def compute_image_quality_score(image_path, threshold=5):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error reading image {image_path}")
        return None
    
    # 对图像进行二维快速傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # 计算幅值谱
    magnitude_spectrum = np.abs(fshift)
    
    # 设置阈值并统计大于该阈值的频率数量
    high_frequency_count = np.sum(magnitude_spectrum > threshold)
    
    return high_frequency_count

def process_folder(folder_path, threshold=5):
    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 仅处理图像文件
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                score = compute_image_quality_score(file_path, threshold)
                if score is not None:
                    print(f'Image: {os.path.basename(file_path)}, Quality Score: {score}')

# 示例文件夹路径
folder_path = '/mnt/lustre/zengziyin/datasets/generated/dreambooth_icfg/eight_id/LLM_text/post_data/yolov8/new_demo'

# 处理文件夹并输出每个图像的质量评分
process_folder(folder_path, threshold=5)
