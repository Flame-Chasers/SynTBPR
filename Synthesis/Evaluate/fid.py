import os
from cleanfid import fid
import glob
# 准备真实数据分布和生成模型的图像数据

cuhk_images_folder = '../datasets/CUHK-PEDES/imgs'
root = '/mnt/lustre/zengziyin/datasets/generated/calc_fid/prompt2'

all_images = ['path_img']
all_images = sorted(all_images)
for generated_images_folder in all_images:
    cuhk_score = fid.compute_fid(cuhk_images_folder, generated_images_folder)
    print('{} Market FID value is {}:'.format(generated_images_folder,cuhk_score))

