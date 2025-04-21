import os
import random
import json
import numpy as np
from diffusers import StableDiffusionPipeline
import time
import torch
import argparse
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion from random descriptions.")

    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_json', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--save_root', type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument('--batch_number', type=int, default=5, help="Number of images to generate per prompt.")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of the entries to process from the JSON file.")
    parser.add_argument('--num_entries', type=int, required=True, help="Number of entries to process from the JSON file.")
    parser.add_argument('--start_img_id', type=int, required=True, help="The starting image ID for generated images.")
    parser.add_argument('--model_id', type=str, required=True, help="Path to the pretrained model.")
    args = parser.parse_args()

    set_seed(42)
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    batch_number = args.batch_number
    save_root = args.save_root
    save_json = args.output_json
    os.makedirs(save_root, exist_ok=True)
    json_root = os.path.dirname(save_json)
    os.makedirs(json_root, exist_ok=True)

    fix_prompt = "sks person"

    # Define the attributes
    all_age = ["young", "middle-aged", "old"]
    all_weather = ["sunny", "rainy", "foggy", "snowy"]
    all_gender = ["female", "male"]
    hair_length = ["short hair", "long hair"]
    height = ["tall", "short", "medium-height"]
    patterns = ["stripe pattern", "cartoon pattern", "geometric patterns", "letters and logo patterns", "photography patterns"]
    type_of_upper_body = ["Denim jacket", "Knit sweater", "T-shirt", "Hoodie", "Blouse", "Tank top", "Polo shirt", "Cardigan", "Button-down shirt", "Tunic", "Vest"]
    sleeve_length = ["long sleeve", "short sleeve"]
    length_of_lower_body = ["long lower body clothing", "short"]
    type_of_lower_body = ["trousers", "skirt", "dress", "pants", "jeans", "shorts", "leggings"]
    color_of_shoes = ['Red', 'Yellow']
    type_of_shoes = ["sports shoes", "casual shoes", "leather shoes", "long boots", "short boots"]
    all_appending = ["holding a cell phone", "wearing glasses", "wearing a mask", "wearing a scarf", "wearing a hat", "carrying a bag"]
    all_angle = ["walking away from the camera", "walking towards the camera", "looking off to the distance", "captured from a security camera angle",
                 "seen from a rooftop", "captured from a corner-mounted camera"]
    # Define the adjectives for lower body
    adjectives_of_lower_body = {
        "trousers": ["tailored", "loose-fitting", "cropped", "wide-leg", "slim-fit"],
        "skirt": ["pleated", "A-line", "maxi", "mini", "wrap", "pencil"],
        "dress": ["flowing", "fitted", "maxi", "midi", "mini", "wrap"],
        "pants": ["casual", "formal", "relaxed-fit", "cargo", "cropped"],
        "jeans": ["distressed", "skinny", "high-waisted", "bootcut", "ripped", "straight-leg"],
        "shorts": ["denim", "Bermuda", "cargo", "high-waisted", "tailored"],
        "leggings": ["stretchy", "athletic", "high-waisted", "patterned", "compression"]
    }

    # Generate random descriptions
    def generate_description():
        age = random.choice(all_age)
        gender = random.choice(all_gender)
        hair = random.choice(hair_length if gender == "female" else ["short hair"])
        sleeve = random.choice(sleeve_length)
        lower_body_options = [item for item in type_of_lower_body if gender == "female" or item not in ["skirt", "dress"]]
        lower_body_type = random.choice(lower_body_options)
        adjective = random.choice(adjectives_of_lower_body[lower_body_type])
        lower = f"{adjective} {lower_body_type}"
        shoes = random.choice(type_of_shoes)
        pattern = random.choice(patterns)
        upper = random.choice(type_of_upper_body)
        angle = random.choice(all_angle)
        appending = random.choice(all_appending)
        return f"A {age} {gender} {fix_prompt}, with {hair}, {pattern} {upper} with {sleeve}, {lower}, a pair of {shoes}, {appending}, {angle}"

    # Generate 100,000 random descriptions
    total_annos = 100000
    annos = [generate_description() for _ in range(total_annos)]

    t2i_dict = []
    begin = time.time()
    begin_idx = args.start_idx
    end_idx = args.end_idx
    idx = args.start_img_id

    for anno in tqdm(annos[begin_idx:end_idx], desc="Generating images"):
        attribute_name = "base"
        random_list = []
        save_path_list = []
        pre_idx = idx
        for i in range(batch_number):
            # idx += 1  
            random_number = random.randint(1, 10000000)
            random_list.append(random_number)
            save_name = f"images_{idx}_{random_number}_{attribute_name}.png"
            save_path = os.path.join(save_root, save_name)
            save_path_list.append(save_path)
            idx += 1  

        generators = [torch.Generator(device="cuda").manual_seed(x) for x in random_list]
        prompt = anno
        images = pipe(prompt, generator=generators, num_images_per_prompt=batch_number, guidance_scale=8.5).images

        idx = pre_idx 
        for i in range(batch_number):
            save_path = save_path_list[i]
            # idx += 1  
            image = images[i]
            t2i_dict.append({'id': idx, 'image_path': save_path, 'caption': prompt})
            image.save(save_path)
            idx += 1  

    with open(save_json, 'w') as json_file:
        json.dump(t2i_dict, json_file, indent=4)

if __name__ == "__main__":
    main()
