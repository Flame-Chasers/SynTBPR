from typing import Union, Tuple, List, Dict
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import DDIMScheduler, DDIMInverseScheduler
from prompt_to_prompt.pipeline_stable_diffusion_nmg import NMGPipeline
from prompt_to_prompt.ptp_utils import (
    AttentionRefine,
    AttentionReplace,
    LocalBlend,
    AttentionReweight,
    get_word_inds,
)
import json
import os
import argparse
import re  

def get_equalizer(text: str,
                  word_select: Union[int, Tuple[int, ...]],
                  values: Union[List[float], Tuple[float, ...]],
                  tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(prompts: List[str],
                    is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float],
                    self_replace_steps: float,
                    blend_word=None,
                    equilizer_params=None,
                    num_steps=None,
                    tokenizer=None,
                    device=None):
    if blend_word is None:
        lb = None
    else:
        lb = LocalBlend(prompts, num_steps, blend_word, tokenizer=tokenizer, device=device)
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_steps, cross_replace_steps=cross_replace_steps, 
                self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device)
    else:
        controller = AttentionRefine(prompts, num_steps, cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=tokenizer)
        controller = AttentionReweight(prompts, num_steps, cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller,
                tokenizer=tokenizer, device=device)
    return controller

NUM_DIFFUSION_STEPS = 50


model_ckpt = "./checkpoints/stable-diffusion-v1-5"
pipe = NMGPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)


pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe.inverse_scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_zero=False)
pipe = pipe.to('cuda')


config = {
    "styles": [
        "In a snowy day",
        "In a rainy day",
        "In a foggy day",
    ],
    "grad_scale": 5000,
    "guidance_noise_map": 10,
    "guidance_text": 10,
    "noise_cond_loss_type": 'l1',
}


def load_json(json_path: str, start_index: int, num_entries: int) -> List[dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[start_index:start_index + num_entries]


def filter_prompts(prompts: List[str], max_length: int = 77) -> List[str]:
    filtered_prompts = [prompt[:max_length] for prompt in prompts]
    return filtered_prompts

def save_image(image: Image, output_folder: str, image_id: int, extracted_number: int, style_index: int):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'images_{image_id}_{extracted_number}_{style_index}.jpg')
    image.save(output_path)
    print(f"Image saved to {output_path}")


def generate_images_from_json(json_path: str, output_folder: str, config: Dict, start_index: int, num_entries: int):
    data = load_json(json_path, start_index, num_entries)
    
    # Progress bar
    for entry in tqdm(data, desc="Generating images", unit="entry"):
        image_id = entry['id']
        src_prompt = entry['captions'][0]
    
        img = Image.open(entry['file_path'])

        extracted_number = int(re.search(r'images_\d+_(\d+)_', entry['file_path']).group(1))

        inv_output = pipe.invert(src_prompt, img, num_inference_steps=NUM_DIFFUSION_STEPS)
        forward_latents = inv_output.latents_list
        

        for idx, style in enumerate(config["styles"], 1):  
            trg_prompt = f"{style}, {src_prompt}"
            prompts = [src_prompt, trg_prompt]
            prompts = filter_prompts(prompts)


            cross_replace_steps = 0.8
            self_replace_steps = 0.6
            blend_word = None
            eq_params = {"words": (style,), "values": (3,)}
            controller = make_controller(prompts,
                                        False,
                                        cross_replace_steps,
                                        self_replace_steps,
                                        blend_word,
                                        eq_params,
                                        NUM_DIFFUSION_STEPS,
                                        pipe.tokenizer,
                                        pipe.device)


            torch.manual_seed(extracted_number)


            with torch.autocast("cuda"): 
                outputs = pipe(
                    prompt=prompts,
                    controller=controller,
                    num_inference_steps=NUM_DIFFUSION_STEPS,
                    grad_scale=config["grad_scale"],
                    guidance_noise_map=config["guidance_noise_map"],
                    guidance_text=config["guidance_text"],
                    noise_cond_loss_type=config["noise_cond_loss_type"],
                    forward_latents=forward_latents  
                )
            

            save_image(outputs.images[1], output_folder, image_id, extracted_number, idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from JSON data with Stable Diffusion.")
    parser.add_argument('--json_path', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder where generated images will be saved.")
    parser.add_argument('--start_index', type=int, default=0, help="Starting index of the JSON entries to process.")
    parser.add_argument('--num_entries', type=int, default=5, help="Number of JSON entries to process.")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_images_from_json(args.json_path, args.output_folder, config, args.start_index, args.num_entries)
