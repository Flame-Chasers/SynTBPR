import os
import random
import json
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
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

def replace_terms(sentence):
    sentence = sentence.replace("person", "sks person")
    sentence = sentence.replace("woman", "female sks person")
    sentence = sentence.replace("man", "male sks person")
    return sentence

def replace_before_comma(text):
    parts = text.split(',', 1)  
    if len(parts) > 1:
        parts[0] = replace_terms(parts[0])
    return ','.join(parts)

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_images_from_json(input_json, output_json, save_root, batch_number, start_idx, num_entries, start_img_id, model_id):
    set_seed(42)

    ensure_dir_exists(save_root)

    json_dir = os.path.dirname(output_json)
    ensure_dir_exists(json_dir)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    with open(input_json, 'r') as f:
        annos = json.load(f)

    t2i_dict = []
    img_id = start_img_id

    end_idx = start_idx + num_entries

    for anno in tqdm(annos[start_idx:end_idx], desc="Generating images"):
        attribute_name = "base"
        random_list = []
        save_path_list = []

        for i in range(batch_number):
            random_number = random.randint(1, 10000000)
            random_list.append(random_number)
            current_img_id = img_id + i
            save_name = f"images_{current_img_id}_{random_number}_{attribute_name}.png"
            save_path = os.path.join(save_root, save_name)
            save_path_list.append(save_path)

        generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in random_list]
        response = anno.get('response', 'No response available')

        if "man" not in response.split(',', 1)[0] and "woman" not in response.split(',', 1)[0] and "person" not in response.split(',', 1)[0]:
            continue

        prompt = replace_before_comma(response)
        images = pipe(prompt, generator=generators, num_images_per_prompt=batch_number, guidance_scale=8.5).images

        for i in range(batch_number):
            images[i].save(save_path_list[i])
            t2i_dict.append({'id': img_id + i, 'image_path': save_path_list[i], 'caption': prompt})

        img_id += batch_number

    with open(output_json, 'w') as out_f:
        json.dump(t2i_dict, out_f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion from a JSON file.")

    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_json', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--save_root', type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument('--batch_number', type=int, default=5, help="Number of images to generate per prompt.")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of the entries to process from the JSON file.")
    parser.add_argument('--num_entries', type=int, required=True, help="Number of entries to process from the JSON file.")
    parser.add_argument('--start_img_id', type=int, required=True, help="The starting image ID for generated images.")
    parser.add_argument('--model_id', type=str, required=True, help="Path to the pretrained model.")
    args = parser.parse_args()

    generate_images_from_json(
        args.input_json,
        args.output_json,
        args.save_root,
        args.batch_number,
        args.start_idx,
        args.num_entries,
        args.start_img_id,
        args.model_id
    )

if __name__ == "__main__":
    main()
