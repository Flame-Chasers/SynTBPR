import torch
from torch import nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm import tqdm
import os
import random
import re

templates = [
    "Wearing [clothing description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Sporting [hair description], the [person/woman/man] is dressed in [clothing description] and is carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "In [clothing description] and [footwear description], the [person/woman/man] is also carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "In [clothing description] and [accessory description], the [person/woman/man] is also carrying [belongings description].",
    "With [hair description], the [person/woman/man] is dressed in [clothing description] and [accessory description].",
    "Sporting [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "In [clothing description] and [accessory description], the [person/woman/man] also has [hair description].",
    "With [accessory description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Wearing [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] sports [hair description] and is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "The [person/woman/man] is attired in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "The [person/woman/man] is dressed in [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description].",
   "The [person/woman/man] is seen wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "Sporting [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "The [person/woman/man] can be spotted wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is dressed in [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is attired in [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description].",
    "Dressed in [accessory description], [clothing description], [footwear description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "Clad in [clothing description] and [footwear description], the [person/woman/man] sports [hair description] and carries [belongings description].",
    "The [person/woman/man] is outfitted in [clothing description] and [accessory description], complemented by [hair description] and carrying [belongings description].",
    "Featuring [hair description], the [person/woman/man] sports [footwear description] and [clothing description], along with [belongings description].",
    "Adorned in [accessory description] and [footwear description], the [person/woman/man] wears [clothing description] and is also carrying [belongings description].",
    "Dressed in [clothing description] and sporting [hair description], the [person/woman/man] holds [belongings description] and wears [footwear description].",
    "The [person/woman/man], with [hair description], is clad in [accessory description] and [clothing description], holding [belongings description].",
    "The [person/woman/man] is equipped with [belongings description], dressed in [clothing description] and [footwear description], with [hair description].",
    "In [clothing description] complemented by [accessory description], the [person/woman/man] also carries [belongings description] and has [hair description].",
    "Decked in [clothing description] and [accessory description], the [person/woman/man] flaunts [hair description] while holding [belongings description].",
    "The [person/woman/man] showcases [hair description] atop [clothing description] with [footwear description], clutching [belongings description].",
    "The [person/woman/man] is spotted with [hair description] while wearing [clothing description] and [footwear description], and toting [belongings description].",
    "Flaunting [hair description], the [person/woman/man] dons [clothing description] and [footwear description], accessorized with [belongings description].",
    "Donning a [clothing description] ensemble with [accessory description], the [person/woman/man] carries [belongings description] and sports [hair description].",
    "The [person/woman/man], wearing [clothing description] and [footwear description], flaunts [hair description] and grips [belongings description].",
    "The [person/woman/man] parades [hair description], attired in [clothing description] and [accessory description], while managing [belongings description].",
    "Garbed in [clothing description] with [footwear description], the [person/woman/man] displays [hair description] and totes [belongings description].",
    "In [clothing description] and [footwear description], the [person/woman/man] with [hair description] is seen clutching [belongings description].",
    "The [person/woman/man] adorned in [clothing description] and [accessory description], balances [belongings description] and styles [hair description].",
    "Wearing [clothing description] topped with [accessory description], the [person/woman/man] bears [hair description] and lugs [belongings description].",
    "Outfitted in [clothing description] and [footwear description], the [person/woman/man] with [hair description] grips [belongings description].",
    "The [person/woman/man] clad in [clothing description], [accessory description], and [footwear description], sports [hair description] while managing [belongings description].",
    "With [hair description] and wearing [clothing description], the [person/woman/man] is accessorized with [belongings description] and [footwear description].",
    "The [person/woman/man] sports [hair description] while decked out in [clothing description], [accessory description], and clutching [belongings description].",
    "In [clothing description] and [accessory description], the [person/woman/man] exhibits [hair description] and juggles [belongings description].",
    "The [person/woman/man], clad in [clothing description] with [hair description], holds [belongings description].",
    "Adorned with [hair description], the [person/woman/man] models [clothing description] and [footwear description], clutching [belongings description].",
    "The [person/woman/man] features [hair description] while sporting [clothing description] and balancing [belongings description].",
    "Clad in [clothing description] and carrying [belongings description], the [person/woman/man] showcases a striking [hair description].",
    "The [person/woman/man], in [clothing description] with [hair description], carries [belongings description].",
    "With [hair description] cascading down, the [person/woman/man] wears [clothing description] and holds [belongings description].",
    "In [clothing description] with [accessory description], the [person/woman/man] exhibits [hair description] and clutches [belongings description].",
    "The [person/woman/man] parades in [clothing description] and [footwear description], crowned with [hair description] and carrying [belongings description].",
    "Clad in a [clothing description] ensemble, the [person/woman/man] with [hair description] elegantly carries [belongings description].",
    "Displaying [hair description], the [person/woman/man] steps out in [clothing description] and [accessory description], with [belongings description].",
    "Wearing [clothing description] and flaunting [hair description], the [person/woman/man] accessorizes with [belongings description].",
    "The [person/woman/man], with [hair description], adorns [clothing description] and [accessory description], carrying [belongings description].",
    "In [clothing description] and [accessory description], the [person/woman/man] showcases [hair description] while holding [belongings description].",
    "The [person/woman/man], showcasing [hair description], is clad in [clothing description] and [accessory description], managing [belongings description].",
    "Wearing [clothing description] and [accessory description], the [person/woman/man] with [hair description] strides forward, holding [belongings description].",
    "The [person/woman/man], draped in [clothing description] with [footwear description], carries [belongings description] and showcases [hair description].",
    "With [hair description], the [person/woman/man] in [clothing description] and [accessory description] juggles [belongings description].",
    "Clad in [clothing description] with [footwear description], the [person/woman/man] sporting [hair description] holds [belongings description].",
    "The [person/woman/man] in [clothing description] and [footwear description], with [hair description], carries [belongings description].",
    "In [clothing description] and carrying [belongings description], the [person/woman/man] flaunts [hair description] and [accessory description].",
    "The [person/woman/man] dons [clothing description] and [footwear description], sporting [hair description], while handling [belongings description].",
    "Wearing [clothing description] and [footwear description], the [person/woman/man] with [hair description] holds [belongings description].",
    "The [person/woman/man], in [clothing description] and [accessory description], showcases [hair description] while clutching [belongings description].",
    "Sporting [hair description], the [person/woman/man] in [footwear description] and [clothing description] carries [belongings description].",
    "Clad in [clothing description], the [person/woman/man] with [hair description] strides along, clutching [belongings description].",
    "The [person/woman/man], carrying [belongings description], flaunts [hair description] while dressed in [clothing description] and [footwear description].",
    "Sporting [hair description], the [person/woman/man] is outfitted in [clothing description] with [accessory description], managing [belongings description].",
    "The [person/woman/man] in [clothing description] and [accessory description], with [hair description], carries [belongings description].",
    "Wearing [clothing description] and [accessory description], the [person/woman/man] displays [hair description] and bears [belongings description].",
    "With [hair description] flowing, the [person/woman/man] models [clothing description] and [footwear description], carrying [belongings description].",
    "The [person/woman/man], in [footwear description] and [clothing description], showcases [hair description] while holding [belongings description].",
    "Adorned in [clothing description], the [person/woman/man] with [hair description] carries [belongings description] and sports [footwear description].",
    "The [person/woman/man], sporting [hair description], steps out in [clothing description] and [footwear description], gripping [belongings description].",
    "Showcasing [hair description], the [person/woman/man] is garbed in [clothing description] and [accessory description], with [belongings description] in hand.",
    "Wearing [clothing description] and [accessory description], the [person/woman/man] with [hair description] carries [belongings description].",
    "With [hair description], the [person/woman/man] in [footwear description] parades in [clothing description], clutching [belongings description].",
    "The [person/woman/man] adorned in [clothing description] with [hair description] manages [belongings description] and wears [footwear description].",
    "Clad in [clothing description] with [hair description], the [person/woman/man] holds [belongings description] while donning [accessory description].",
    "Wearing [clothing description], [footwear description], and sporting [hair description], the [person/woman/man] also carries [belongings description].",
    "In [clothing description] and [accessory description], the [person/woman/man] with [hair description] strides while carrying [belongings description].",
    "Garbed in [clothing description], the [person/woman/man] with [hair description] balances [belongings description].",
    "The [person/woman/man], adorned in [clothing description] and [accessory description], carries [belongings description] with a flair of [hair description].",
    "With [hair description] perfectly styled, the [person/woman/man] showcases [clothing description] and [footwear description], handling [belongings description].",
    "Carrying [belongings description], the [person/woman/man] in [clothing description] steps out, complemented by [accessory description] and [hair description].",
    "The [person/woman/man] in [clothing description] and [footwear description], flaunts [hair description] while gripping [belongings description].",
    "Dressed in [clothing description] with [hair description], the [person/woman/man] maneuvers [belongings description] and adds a touch of [accessory description].",
    "The [person/woman/man], showcasing [hair description], is attired in [clothing description] with [accessory description], clutching [belongings description].",
    "In [clothing description] and [hair description], the [person/woman/man] carries [belongings description] while sporting [footwear description].",
    "The [person/woman/man] wearing [clothing description] and [footwear description] displays [hair description] and holds [belongings description].",
    "The [person/woman/man] in [clothing description], [footwear description], and [accessory description] strides forward, sporting [hair description] and [belongings description].",
    "Clad in [clothing description], the [person/woman/man] with [hair description] showcases [footwear description] and manages [belongings description].",
    "The [person/woman/man], in [clothing description] and [hair description], walks with [belongings description] and [accessory description].",
    "Wearing [clothing description], the [person/woman/man] with [hair description] carries [belongings description] and [footwear description].",
    "The [person/woman/man] sporting [hair description] and [clothing description] holds [belongings description] while wearing [footwear description].",
    "In [clothing description] and [accessory description], the [person/woman/man] with [hair description] handles [belongings description].",
    "The [person/woman/man] in [clothing description] and [footwear description] displays [hair description] and carries [belongings description].",
    "Sporting [hair description], the [person/woman/man] dressed in [clothing description] and [accessory description] holds [belongings description].",
    "The [person/woman/man], with [hair description], steps out in [clothing description] and [footwear description], carrying [belongings description].",
    "Clad in [clothing description] with [hair description], the [person/woman/man] manages [belongings description] while flaunting [accessory description].",
    "Sporting [hair description], the [person/woman/man] clad in [clothing description] with [accessory description] manages [belongings description].",
    "The [person/woman/man], adorned in [clothing description] and [footwear description], sports [hair description] while toting [belongings description].",
    "With [hair description] styled, the [person/woman/man] showcases a [clothing description] complemented by [belongings description].",
    "The [person/woman/man] in [clothing description], adorned with [accessory description], carries [belongings description] and displays [hair description].",
    "Decked in [clothing description] and [footwear description], the [person/woman/man] with [hair description] holds [belongings description].",
    "Wearing [clothing description] and [hair description], the [person/woman/man] handles [belongings description] with [accessory description] in tow.",
    "The [person/woman/man], showcasing [hair description], strides in [clothing description] while clutching [belongings description] and [footwear description].",
    "In [clothing description] with [hair description], the [person/woman/man] carries [belongings description], accentuated by [footwear description].",
    "The [person/woman/man] sporting [hair description] and [clothing description] carries [belongings description], dressed in [footwear description].",
    "Clad in [clothing description] with [hair description], the [person/woman/man] strides forward, carrying [belongings description] and sporting [accessory description].",
    "The [person/woman/man] in [clothing description] and [footwear description], with [hair description], holds [belongings description].",
    "Wearing [clothing description], the [person/woman/man] with [hair description] manages [belongings description] and [footwear description].",
    "The [person/woman/man], flaunting [hair description], is attired in [clothing description] and [accessory description], holding [belongings description].",
    "In [clothing description] and [footwear description], the [person/woman/man] with [hair description] juggles [belongings description].",
    "The [person/woman/man], sporting [hair description], dons [clothing description] and carries [belongings description] along with [footwear description].",
    "The [person/woman/man] in [clothing description] and [hair description], carries [belongings description] while showcasing [footwear description].",
    "The [person/woman/man], dressed in [clothing description] with [hair description], carries [belongings description] and [footwear description].",
    "In [clothing description], the [person/woman/man] with [hair description] swings [belongings description] while stepping in [footwear description].",
    "The [person/woman/man], garbed in [clothing description] with [hair description], clasps [belongings description] and accents with [accessory description].",
    "Wearing [clothing description] and [accessory description], the [person/woman/man] with [hair description] strides while holding [belongings description].",
    "The [person/woman/man], adorned in [clothing description] and [hair description], carries [belongings description] with a [accessory description].",
    "Decked in [clothing description], the [person/woman/man] sporting [hair description] maneuvers [belongings description] while showcasing [footwear description].",
    "With [hair description] and wearing [clothing description], the [person/woman/man] handles [belongings description] alongside [footwear description].",
    "The [person/woman/man], in [clothing description] and [accessory description], showcases [hair description] while juggling [belongings description].",
    "Flaunting [hair description], the [person/woman/man] in [clothing description] balances [belongings description] and [footwear description] with ease.",
    "Clad in [clothing description] with [footwear description], the [person/woman/man] with [hair description] carries [belongings description].",
    "The [person/woman/man], donning [clothing description] and [hair description], totes [belongings description] and wears [accessory description].",
    "In [clothing description] and [footwear description], the [person/woman/man] adorned with [hair description] handles [belongings description] gracefully.",
    "Wearing [clothing description], the [person/woman/man] with [hair description] carries [belongings description] and sports [footwear description] elegantly.",
    "The [person/woman/man], showcasing [hair description] and [clothing description], manages [belongings description] and [accessory description].",
    "With [hair description], the [person/woman/man] in [clothing description] and [accessory description] strides forward carrying [belongings description].",
    "Dressed in [clothing description] with [hair description], the [person/woman/man] carries [belongings description] and shows off [accessory description].",
    "The [person/woman/man], sporting [hair description] and [footwear description], wears [clothing description] while clutching [belongings description].",
    "The [person/woman/man], dressed in [clothing description] with [hair description], manages [belongings description] and [footwear description].",
    "Clad in [clothing description] and flaunting [hair description], the [person/woman/man] navigates [belongings description] with [footwear description] on display.",
    "The [person/woman/man], adorned with [hair description], drapes [clothing description] while managing [belongings description] and a subtle [accessory description].",
    "Sporting [hair description], the [person/woman/man] in [clothing description] strides , toting [belongings description] and complemented by [footwear description].",
    "With [hair description] and dressed in [clothing description], the [person/woman/man] carries [belongings description] and accentuates their look with [accessory description]."
]  

def generate_question():
    template = random.choice(templates)
    question = f'Generate a description about the overall appearance of the person, including the clothing, shoes, hairstyle, gender and belongings, in a style similar to the template:"{template}". If some requirements in the template are not visible, you can ignore. Do not imagine any contents that are not in the image. Your answer should not mention the style of the image or the context in which the person is placed in the image.'
    return question

def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))

def run(args):
    model = AutoModel.from_pretrained(
        './checkpoints/MiniCPM-V-2_6', # You can download the model from "https://huggingface.co/openbmb/MiniCPM-V-2_6" 
        trust_remote_code=True,
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16) 
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        './checkpoints/MiniCPM-V-2_6', # You can download the model from "https://huggingface.co/openbmb/MiniCPM-V-2_6" 
        trust_remote_code=True
    )

    all_imgs = sorted(
        [img for img in os.listdir(args.image_root) if img.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=extract_numbers
    )
    
    selected_imgs = all_imgs[args.start_index:args.start_index + args.num_images]
    
    output_data = []
    
    for img_path in tqdm(selected_imgs, desc="Processing images"):
        absolute_image_path = os.path.abspath(os.path.join(args.image_root, img_path))
        image = Image.open(absolute_image_path).convert('RGB')
        
        question = generate_question()
        msgs = [{'role': 'user', 'content': [image, question]}]
        
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,
            stream=False
        )
        
        generated_text = "".join(res)
        
        entry = {
            "image_path": absolute_image_path,
            "captions": [generated_text],
            "MiniCPM_question": question,
            "split": "train"
        }
        output_data.append(entry)
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Processing completed. Results saved to '{args.output_json}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use the MiniCPM-V-2_6 model to process images and generate descriptions.')
    parser.add_argument('--image_root', type=str, required=True, help='Path to the directory containing the images')
    parser.add_argument('--output_json', type=str, required=True, help='The path where the output JSON file will be saved.')
    parser.add_argument('--start_index', type=int, required=True, help='The index of the first image to process (starting from 0)')
    parser.add_argument('--num_images', type=int, required=True, help='Number of images to process')
    args = parser.parse_args()

    run(args)
