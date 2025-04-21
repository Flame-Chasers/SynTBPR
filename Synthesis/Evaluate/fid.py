import os
from cleanfid import fid
import glob
# 准备真实数据分布和生成模型的图像数据
# /mnt/lustre/zengziyin/datasets/generated/calc_fid/cuhk-test
# /mnt/lustre/zengziyin/datasets/CUHK-PEDES/imgs
cuhk_images_folder = '/mnt/lustre/zengziyin/datasets/generated/calc_fid/cuhk-test'
root = '/mnt/lustre/zengziyin/datasets/generated/calc_fid/prompt2'
all_images = glob.glob(os.path.join(root,'*'))
all_images = ['/mnt/lustre/zengziyin/datasets/generated/zero-shot/sd1.5/cuhk-pedes/base_eval/post_data/yolov8/calc_fid']
all_images = sorted(all_images)
for generated_images_folder in all_images:
    # generated_images_folder = '/mnt/lustre/zengziyin/datasets/generated/dreambooth_cuhk03/same_id/process_data/ori1_resize_0.5hflip_sks_500_2.0e-6_1.0loss'

    # score = fid.compute_fid(real_images_folder, generated_images_folder)
    # print('FID value:', score)
    # print("generated_images_folder = {}".format(generated_images_folder))
    cuhk_score = fid.compute_fid(cuhk_images_folder, generated_images_folder)
    print('{} Market FID value is {}:'.format(generated_images_folder,cuhk_score))

