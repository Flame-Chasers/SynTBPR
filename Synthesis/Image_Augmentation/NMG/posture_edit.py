import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.io import read_image
from pytorch_lightning import seed_everything
from diffusers import DDIMScheduler

from MasaCtrl.diffuser_utils import MasaCtrlPipeline
from MasaCtrl.masactrl_utils import regiter_attention_editor_diffusers
from MasaCtrl.masactrl import MutualSelfAttentionControl

import json
import os
from tqdm import tqdm  

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def main(args):

    json_path = args.json_path

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = "./checkpoints/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

    seed = 42
    seed_everything(seed)

    with open(json_path, 'r') as f:
        data = json.load(f)

    trg_prompts = [
        "a person, sitting",
        "a person, raising hands",
        "a person, giving a thumb up",
        "a person, running",
        "a person, walking",
    ]


    start_index = args.start_index
    end_index = args.end_index

    for item in tqdm(data[start_index:end_index], desc="Processing data"):
        try:
            for i, trg_prompt in enumerate(trg_prompts, 1):
                file_path = item['file_path']


                original_filename = os.path.basename(file_path)
                name, ext = os.path.splitext(original_filename)
                if name.endswith('_base'):
                    name = name[:-5]  
                elif name.endswith('base'):
                    name = name[:-4]  

                img = load_image(file_path, device)

                src_prompt = "a person"
                prompts = [src_prompt, trg_prompt]

                start_code, latents_list = model.invert(
                    img,
                    src_prompt,
                    guidance_scale=1,
                    num_inference_steps=50,
                    return_intermediates=True
                )
                start_code = start_code.expand(2, -1, -1, -1)

                STEP = 4
                LAYER = 10

                editor = MutualSelfAttentionControl(STEP, LAYER)
                regiter_attention_editor_diffusers(model, editor)
                grad_scale = 5000          
                guidance_noise_map = 10    
                guidance_text = 10         

                image_masactrl = model(
                    prompts,
                    latents=start_code,
                    guidance_scale=guidance_text,
                    ref_intermediate_latents=latents_list,
                    grad_scale=grad_scale,
                    guidance_noise_map=guidance_noise_map
                )

                gen_img = image_masactrl[1]

                output_filename = f"{name}_{i}{ext}"
                output_path = os.path.join(output_dir, output_filename)


                pil_image = ToPILImage()(gen_img.cpu())
                pil_image.save(output_path)


        except Exception as e:
            print(f"Process {file_path} Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on JSON data.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of JSON data to process.")
    parser.add_argument("--end_index", type=int, default=50, help="End index of JSON data to process.")
    args = parser.parse_args()
    main(args)
