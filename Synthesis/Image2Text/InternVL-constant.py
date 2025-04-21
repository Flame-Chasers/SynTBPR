import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import re

parser = argparse.ArgumentParser(description="Use InternVL to process images and generate descriptions")
parser.add_argument("--image_root", type=str, required=True, help="Path to the directory containing the images")
parser.add_argument("--output_json", type=str, required=True, help="The path to save the output JSON file")
parser.add_argument("--start_index", type=int, required=True, help="The index of the first image to process (starting from 0)")
parser.add_argument("--num_images", type=int, required=True, help="Number of images to process")
args = parser.parse_args()

output_dir = os.path.dirname(args.output_json)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def static_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = static_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "./checkpoints/InternVL-Chat-V1-5"    # You can download this model from "https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5" 
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

def generate_response(pixel_values, question):
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))

all_imgs = sorted(
    [img for img in os.listdir(args.image_root) if img.lower().endswith(('.jpg', '.jpeg', '.png'))],
    key=extract_numbers
)
selected_imgs = all_imgs[args.start_index:args.start_index + args.num_images]

output_data = []
question = "Don't mention the background environment of the people in the image. Please provide a detailed description of this person's age, gender, top (including color and style), bottom (including color and style), hair (including color and style), shoes (including color and style), and belongings (including color and style). Finally, combine all the details into a single sentence."

for img_path in tqdm(selected_imgs, desc="Processing Images"):
    image_path = os.path.join(args.image_root, img_path)
    absolute_image_path = os.path.abspath(image_path)

    pixel_values = load_image(absolute_image_path, max_num=6).to(torch.bfloat16).cuda()
    response = generate_response(pixel_values, question)
    
    output_data.append({
        "image_path": absolute_image_path,
        "captions": [response],
        "split":"train",
        "InternVL_question":question
    })

with open(args.output_json, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Results saved to {args.output_json}")
