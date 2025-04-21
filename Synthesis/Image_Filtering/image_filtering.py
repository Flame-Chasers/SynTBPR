import glob
import os
import torch
import numpy as np
import random
import shutil
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import re

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

def judge(keypoint):
    for x, y in keypoint:
        if x > 0.0001 and y > 0.00001:
            return True
    return False

# Helper function to extract the number for sorting
def extract_number_from_filename(filename):
    match = re.search(r'images_(\d+)_', filename)
    return int(match.group(1)) if match else 0

def filter(image_folder, save_root, start_idx, num_entries):

    model = YOLO('./checkpoints/yolov8/yolov8x-pose-p6.pt')

    os.makedirs(save_root, exist_ok=True)

    all_save_path = []

    image_paths = glob.glob(os.path.join(image_folder, "*.png")) + glob.glob(os.path.join(image_folder, "*.jpg"))


    image_paths = sorted(image_paths, key=extract_number_from_filename)


    end_idx = start_idx + num_entries
    if start_idx < 0 or start_idx >= len(image_paths):
        raise ValueError("Invalid start index")
    if end_idx > len(image_paths):
        end_idx = len(image_paths)

    image_paths = image_paths[start_idx:end_idx]


    for image_path in tqdm(image_paths, desc="Processing Images"):
        results = model(image_path, save=True, save_crop=True, classes=[0], iou=0.1)
        for result in results:
            keypoints = result.keypoints
            if keypoints.shape[0] != 1:
                print(f"{image_path} is not one person")
                continue
            keypoint = keypoints.xy[0]
            if judge(keypoint[:5]) and judge(keypoint[5:11]) and judge(keypoint[13:]):
                image_name = os.path.basename(image_path).replace('.png', '.jpg')
                pre_save_path = os.path.join(result.save_dir, 'crops', 'person', image_name)
                post_save_path = os.path.join(save_root, image_name)
                shutil.copy2(pre_save_path, post_save_path)
                all_save_path.append(pre_save_path)
            else:
                print(f"{image_path} is not one person")

    print(f"save path len = {len(all_save_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with YOLOv8 and save results.")
    parser.add_argument('--image_folder', type=str, required=True, help="Directory containing images to process")
    parser.add_argument('--save_root', type=str, required=True, help="Directory to save processed images")
    parser.add_argument('--start_idx', type=int, required=True, help="Starting index of images to process")
    parser.add_argument('--num_entries', type=int, required=True, help="Number of images to process")

    args = parser.parse_args()

    filter(args.image_folder, args.save_root, args.start_idx, args.num_entries)
