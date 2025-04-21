import json
import os
import random
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm  
import argparse 


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# You can set seed here
set_seed(42)

# You can download the model from the following link：https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
model_id = "./checkpoints/Text_to_Image/S1/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None  
pipe.requires_safety_checker = False

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_images_from_json(input_json, output_json, save_root, batch_number):

    ensure_dir_exists(save_root)

    json_dir = os.path.dirname(output_json)
    ensure_dir_exists(json_dir)
    with open(input_json, 'r') as f:
        annos = json.load(f)
    
    t2i_dict = []  
    img_id = 0  

    for anno in tqdm(annos, desc="Generating images"):
        attribute_name = "base"  
        random_list = []
        save_path_list = []

        for i in range(batch_number):
            random_number = random.randint(1, 10000000)
            random_list.append(random_number)
            save_name = "images_" + str(img_id) + "_" + str(random_number) + "_" + attribute_name + ".png"
            save_path = os.path.join(save_root, save_name)
            save_path_list.append(save_path)
            img_id += 1 

        generator = [torch.Generator(device="cuda").manual_seed(x) for x in random_list]
        prompt = anno['response']
        images = pipe(prompt, generator=generator, num_images_per_prompt=batch_number, guidance_scale=8.5).images

        for i in range(batch_number):
            save_path = save_path_list[i]
            images[i].save(save_path)
            t2i_dict.append({'id': img_id - batch_number + i, 'image_path': save_path, 'caption': prompt})

    with open(output_json, 'w') as out_f:
        json.dump(t2i_dict, out_f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion from a JSON file.")
    
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_json', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--save_root', type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument('--batch_number', type=int, default=5, help="Number of images to generate per prompt.")
    
    args = parser.parse_args()
    
    generate_images_from_json(args.input_json, args.output_json, args.save_root, args.batch_number)

if __name__ == "__main__":
    main()
