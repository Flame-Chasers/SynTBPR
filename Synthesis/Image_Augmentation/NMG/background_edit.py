import argparse
from typing import Union, Tuple, List, Dict
from PIL import Image
import torch
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
from tqdm import tqdm 
from torchvision.transforms import ToPILImage

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

def main(args):
    # Load stable diffusion
    model_ckpt = "./checkpoints/stable-diffusion-v1-5"
    pipe = NMGPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16, safety_checker=None)

    # Set scheduler and inverse scheduler
    pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe.inverse_scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_zero=False)
    pipe = pipe.to('cuda')

    # Filter the locations to single-word items
    locations_scenes = ["forest", "desert", "beach", "airport", "farm", "lake"]

    # JSON file path
    json_path = args.json_path  
    # Output directory for generated images
    output_dir = args.output_dir  
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Process the selected data range
    start_index = args.start_index
    end_index = args.end_index

    for item in tqdm(data[start_index:end_index], desc="Processing data"):
        try:
            file_path = item['file_path']
            original_filename = os.path.basename(file_path)
            name, ext = os.path.splitext(original_filename)
            
            # Remove '_base' from the original filename if present
            if name.endswith('_base'):
                name = name[:-5]  # Remove '_base'
            elif name.endswith('base'):
                name = name[:-4]  # Remove 'base'

            img = Image.open(file_path)

            # Set source prompt
            src_prompt = "a person in the street"
            inv_output = pipe.invert(src_prompt, img, num_inference_steps=NUM_DIFFUSION_STEPS)
            forward_latents = inv_output.latents_list
            
            # Loop through each location to generate the images
            for idx, location in enumerate(locations_scenes, 1):
                # Set target prompt
                trg_prompt = f"a person in the {location}"
                prompts = [src_prompt, trg_prompt]

                # Inversion


                # Set prompt-to-prompt parameters
                cross_replace_steps = 0.8
                self_replace_steps = 0.5
                src_text = "street"
                trg_text = location
                blend_word = (((src_text,), (trg_text,)))
                eq_params = {"words": (trg_text,), "values": (2,)}
                controller = make_controller(prompts,
                                             True,
                                             cross_replace_steps,
                                             self_replace_steps,
                                             blend_word,
                                             eq_params,
                                             NUM_DIFFUSION_STEPS,
                                             pipe.tokenizer,
                                             pipe.device)

                # Set NMG parameters
                grad_scale = 5000  # Gradient scale
                guidance_noise_map = 10  # NMG scale
                guidance_text = 10  # CFG scale
                noise_cond_loss_type = 'l1'  # Choices=['l1', 'l2', 'smooth_l1']

                # NMG with prompt-to-prompt
                with torch.autocast("cuda"):
                    outputs = pipe(
                        prompt=prompts,
                        controller=controller,
                        num_inference_steps=NUM_DIFFUSION_STEPS,
                        grad_scale=grad_scale,
                        guidance_noise_map=guidance_noise_map,
                        guidance_text=guidance_text,
                        noise_cond_loss_type=noise_cond_loss_type,
                        forward_latents=forward_latents
                    )

                # Save the output image with the location name and index (1-6) as the suffix
                output_filename = f"{name}_{idx}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                outputs.images[1].save(output_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on JSON data.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of JSON data to process.")
    parser.add_argument("--end_index", type=int, default=50, help="End index of JSON data to process.")
    args = parser.parse_args()
    main(args)
