# An Empirical Study of Validating Synthetic Data for Text-Based Person Retrieval


## Environment

All the experiments are conducted on either 4 Nvidia RTX 3090 (24GB) GPUs or 1 Nvidia H800 (80GB) GPU. The CUDA version is 11.8.

The required packages are listed in `environment.yml`. You can install them using:

```sh
conda env create -f environment.yml
```

## Download
1. Download CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset).
2. Download the annotation json files from [here](https://drive.google.com/file/d/1C5bgGCABtuzZMaa2n4Sc0qclUvZ-mqG9/view?usp=drive_link).
3. Download the pretrained CLIP checkpoint from [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).


## Training


### 🚀 Training with Generated Data

You can easily start training using **generated data** with PyTorch's `torchrun`.

* If you're using **a single GPU (e.g., 80G)**, only one process is needed and you can set a **larger batch size**.
* If you're using **multiple GPUs**, launch multiple processes and adjust the batch size accordingly.

#### ✅ Single-GPU Training (e.g., A100 80G)

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 \
main.py
```

In `config/config.yaml`, set:

```yaml
batch_size: 320
```

#### ✅ Multi-GPU Training (e.g., 4× A100 40G)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=4 \
main.py 
```

In `config/config.yaml`, set:

```yaml
batch_size: 80
```

---

### 📦 Training with Real Data

To train on **real data**, just add the `--real_data` flag. The same single-GPU / multi-GPU logic applies.

#### ✅ Single-GPU Training with Real Data

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 \
main.py --real_data
```

In `config/real_data_config.yaml`, set:

```yaml
batch_size: 320
```

#### ✅ Multi-GPU Training with Real Data

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=4 \
main.py --real_data
```

In `config/real_data_config.yaml`, set:

```yaml
batch_size: 80
```

---

### 💡 Notes

* The `batch_size` here refers to the **per-GPU batch size**.







## Acknowledgement
+ [TBPS-CLIP](https://arxiv.org/abs/2308.10045) The retrieval model is based on TBPS-CLIP.
+ [EVA-CLIP](https://arxiv.org/abs/2303.15389) The model backbone has been modified to EVA-CLIP.

