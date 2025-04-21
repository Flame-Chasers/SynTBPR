# An Empirical Study of Validating Synthetic Data for Text-Based Person Retrieval

## 🚀 Run

### 1. Text to Image

#### Environment

First you need to create a separate environment.

```conda
conda create -n StableDiffusion python=3.10
```

Then You only need to follow the tutorial on this website: [Diffusers Installation](https://huggingface.co/docs/diffusers/main/en/installation) to install 'diffusers'. 

Lastly，type in the command line to download 'tqdm' to help clearly show the code running process.

```bash
pip install tqdm
```

#### Download

Download the model from the following link：[stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

Place the downloaded checkpoints into the folder "./checkpoints/Text_to_Image/S1/stable-diffusion-v1-5"

#### Scenario-1: no data available

```bash
conda activate StableDiffusion
python /Text_to_Image/Text_to_Image_S1.py \
	--input_json path/to/input.json \
	--output_json path/to/output.json \
	--save_root path/to/save/directory \
	--batch_number Number of images to generate per prompt
```

#### Scenario-2: limit data

As seen in our paper, we developa plain description template (P. TPL), and four rough description templates(R. TMPL) focusing on appearance, profession, location and state. We also use DreamBooth to fine-tune on the CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets. Moreover, we conducted an ablation experiment on the number of IDs in this part.

Therefore, ‘location’, 'CUHK-PEDES', ‘8id’ are selected as an example here.

The ‘model_id' can be downloaded from <mark>here</mark>, and please place checkpoints into the folder "./checkpoints/Text_to_Image/S2"

```bash
conda activate StableDiffusion
python /Text_to_Image/Text_to_Image_S2_location.py \
    --input_json path/to/input.json \
    --output_json path/to/output.json \
    --save_root path/to/save/directory \
    --batch_number Number of images to generate per prompt \
    --start_idx Start index of the entries to process from the JSON file \
    --num_entries Number of entries to process from the JSON file \
    --start_img_id The starting image ID for generated images \
    --model_id "checkpoints/Text_to_Image/S2/dreambooth_ckts/sd1.5/cuhk-pedes/train_query_ckpt/4id"
```

#### Scenario-3: abundance of data

You only need to change the input of model_id here to use the checkpoints of the three datasets CUHK-PEDES, ICFG-PEDES, and RSTPReid in the S3 stage.

The ‘model_id' can be downloaded from here, and please place checkpoints into the folder "./checkpoints/Text_to_Image/S3"

Here we take the CUHK-PEDES dataset as an example.

```bash
conda activate StableDiffusion
python Text_To_Image_S3.py \
	--generate_len Number of entries to generate \
	--Multiplier Multiplier for generating begin_idx \
	--generate_name Name for generated data \
	--model_id "./checkpoints/Text_to_Image/S3/advance_dreambooth_lora/sd1.5/cuhk/ssm/step4000_rank8_bs64"
```

### 2. Image filtering

#### Environment

You can create an environment named 'Yolo' following the instruction [here](https://github.com/ultralytics/ultralytics?tab=readme-ov-file).

You can download the checkpoints here and place it in the foler "./checkpoints/yolov8/yolov8x-pose-p6.pt"

Then, type in the command line to start to filter useless images.

```bash
conda activate Yolo
python image_filtering.py \
    --image_folder /path/to/images \
    --save_root /path/to/save_processed_images \
    --start_idx Starting index of images to process \
    --num_entries Number of images to process
```

### 3. Image to Text

#### （1）InternVL

##### Environment

You can create an environment named 'InternVL' following the instruction [here](https://github.com/OpenGVLab/InternVL).

##### Download

You can download the checkpoints [here](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) and place it in the foler "./checkpoints/InternVL"

Then, type in the command line to start to generate constant text.

```bash
conda activate InternVl
python InternVL-constant.py \
    --image_root /path/to/image_directory \
    --output_json /path/to/output.json \
    --start_index The index of the first image to process \
    --num_images Number of images to process
```

Type in the command line to start to generate variable text.

```bash
conda activate InternVl
python InternVL-variable.py \
    --image_root /path/to/image_directory \
    --output_json /path/to/output.json \
    --start_index The index of the first image to process \
    --num_images Number of images to process
```

#### （2）QwenVL

##### Environment

You can create an environment named 'QwenVL' following the instruction [here](https://github.com/QwenLM/Qwen-VL).

##### Download

You can download the checkpoints [here](https://huggingface.co/Qwen/Qwen-VL-Chat) and place it in the foler "./checkpoints/QwenVL"

Type in the command line to start to generate variable text.

```bash
conda activate InternVl
python QwenVL-variable.py \
    --image_root /path/to/image_directory \
    --output_json /path/to/output.json \
    --start_index The index of the first image to process \
    --num_images Number of images to process
```

#### （3）MiniCPM

##### Environment

Clone the repository from [here](https://github.com/OpenBMB/MiniCPM) to /Image to Text/MiniCPM, and create an environment named MiniCPM according to the instructions [here](https://github.com/OpenBMB/MiniCPM).

##### Download

You can download the checkpoints [here](https://huggingface.co/openbmb/MiniCPM-V-2_6) and place it in the foler "./checkpoints/MiniCPM"

Type in the command line to start to generate variable text.

```bash
conda activate MiniCPM
python MiniCPM-variable.py \
    --image_root /path/to/image_directory \
    --output_json /path/to/output.json \
    --start_index The index of the first image to process \
    --num_images Number of images to process
```

### 4. Image augmentation

#### Environment

Clone the repository from [here](https://github.com/hansam95/NMG) to "./Image augmentation/NMG", and create an environment named NMG according to the instructions [here](https://github.com/hansam95/NMG).

The four attributes we chose to edit are background, weather, style, and posture.

Take background editing as an example, type in the command line to start editing.

```bash
python your_script.py \
    --json_path /path/to/input.json \
    --output_dir /path/to/output_directory \
    --start_index Start index of JSON data to process \
    --end_index End index of JSON data to process
```

### 5. Baseline Model

We utilize the simplified version of [TBPS-CLIP](https://github.com/Flame-Chasers/TBPS-CLIP) with slight modifications as our baseline model.

We call our baseline model as "TBPS-EVACLIP" and you can find it in the folder "Retrieval", see the [here](../Retrieval/) file.


















