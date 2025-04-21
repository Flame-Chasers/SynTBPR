import argparse
import json
import os
import random
import time

import numpy as np
from tqdm import tqdm
from diffusers import AutoPipelineForText2Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with a pipeline.")
    parser.add_argument('--generate_len', type=int, required=True, help="Number of entries to generate")
    parser.add_argument('--Multiplier', type=int, required=True, help="Multiplier for generating begin_idx")
    parser.add_argument('--generate_name', type=str, required=True, help="Name for generated data")
    parser.add_argument('--model_id', type=str, required=True, help="Path to the model checkpoint")
    
    return parser.parse_args()

args = parse_args()

set_seed(42)  

pipe = AutoPipelineForText2Image.from_pretrained('./checkpoints/stable-diffusion-v1-5', torch_dtype=torch.float16).to('cuda')
pipe.load_lora_weights(args.model_id, weight_name='pytorch_lora_weights.safetensors')

embedding_path = f'{args.model_id}/save_emb.safetensors'
state_dict = load_file(embedding_path)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

pipe.safety_checker = None
pipe.requires_safety_checker = False
batch_number = 5

root = "./datasets/generated/all_data/dreambooth_lora/cuhk-pedes/LLM_text/process_data"
json_root = "./datasets/generated/all_data/dreambooth_lora/cuhk-pedes/LLM_text/prompt"
save_root = os.path.join(root, args.generate_name)
save_json = os.path.join(json_root, args.generate_name + ".json")
os.makedirs(save_root, exist_ok=True)
os.makedirs(json_root, exist_ok=True)

json_path = './datasets/CUHK-PEDES/processed_data/train_reid.json'
with open(json_path, "r") as f:
    annos = json.load(f)

t2i_dict = []
begin = time.time()
generate_len = args.generate_len

begin_idx = args.generate_len * args.Multiplier
end_idx = begin_idx + generate_len
idx = begin_idx * batch_number

for anno in tqdm(annos[begin_idx:end_idx], desc="Generating images"):
    attribute_name = "base"
    random_list = []
    save_path_list = []
    pre_idx = idx
    for i in range(batch_number):
        idx += 1
        random_number = random.randint(1, 10000000)
        random_list.append(random_number)
        save_name = "images_" + str(idx) + "_" + str(random_number) + "_" + attribute_name + ".png"
        save_path = os.path.join(save_root, save_name)
        save_path_list.append(save_path)
    
    generator = [torch.Generator(device="cuda").manual_seed(x) for x in random_list]
    response = anno["captions"][0]
    prompt = "in the style of <s0><s1>," + response
    images = pipe(prompt, generator=generator, num_images_per_prompt=batch_number, guidance_scale=8.5).images
    
    idx = pre_idx
    for i in range(batch_number):
        save_path = save_path_list[i]
        idx += 1
        image = images[i]
        t2i_dict.append({'id': idx, 'image_path': save_path, 'caption': prompt})
        image.save(save_path)

end_time = time.time()
print("span time = {}  time of per anno = {}".format(end_time - begin, (end_time - begin) / generate_len))
print("save_json = {}".format(save_json))

with open(save_json, 'w') as json_file:
    json.dump(t2i_dict, json_file, indent=4)
