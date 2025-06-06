import os
import random
import json
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import time
import argparse
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def replace_terms(sentence):
    sentence = sentence.replace('person', 'sks person')
    sentence = sentence.replace('woman', 'female sks person')
    sentence = sentence.replace('man', 'male sks person')
    return sentence

def replace_before_comma(text):
    parts = text.split(',', 1)  # Split at the first comma
    if len(parts) > 1:
        parts[0] = replace_terms(parts[0])
    else:
        parts[0] = replace_terms(parts[0])
    return ','.join(parts)

def main():
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion from a JSON file.')

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
    pipe = pipe.to('cuda')
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    batch_number = args.batch_number
    save_root = args.save_root
    save_json = args.output_json
    os.makedirs(save_root, exist_ok=True)
    json_root = os.path.dirname(save_json)
    os.makedirs(json_root, exist_ok=True)


    json_path = args.input_json
    with open(json_path, 'r') as file:
        annos = json.load(file)

    t2i_dict = []
    begin = time.time()
    generate_len = args.num_entries
    begin_idx = args.start_idx
    end_idx = begin_idx + generate_len
    idx = args.start_img_id

    for anno in tqdm(annos[begin_idx:end_idx], desc="Generating images"):
        attribute_name = 'base'
        random_list = []
        save_path_list = []
        pre_idx = idx
        for i in range(batch_number):

            random_number = random.randint(1, 10000000)
            random_list.append(random_number)
            save_name = f'images_{idx}_{random_number}_{attribute_name}.png'
            save_path = os.path.join(save_root, save_name)
            save_path_list.append(save_path)
            idx += 1  

        generators = [torch.Generator(device='cuda').manual_seed(x) for x in random_list]
        response = anno.get('response', 'No response available')
        if 'man' not in response.split(',', 1)[0] and 'woman' not in response.split(',', 1)[0] and 'person' not in response.split(',', 1)[0]:
            continue
        prompt = replace_before_comma(response)
        images = pipe(prompt, generator=generators, num_images_per_prompt=batch_number, guidance_scale=8.5).images

        idx = pre_idx  
        for i in range(batch_number):
            save_path = save_path_list[i]
            # idx += 1  
            image = images[i]
            t2i_dict.append({'id': idx, 'image_path': save_path, 'caption': prompt})
            image.save(save_path)
            idx += 1  

    end_time = time.time()
    print(f'span time = {end_time - begin}  time per anno = {(end_time - begin) / len(annos)}')
    print(f'save_json = {save_json}')
    with open(save_json, 'w') as json_file:
        json.dump(t2i_dict, json_file, indent=4)

if __name__ == '__main__':
    main()
