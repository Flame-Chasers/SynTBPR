from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNorm, QuickGELU, TextTransformer
from model.tokenizer import tokenize
from .eda import EDA
from misc.utils import is_using_distributed
from .shared_modules import AllGather


try:
    from apex.normalization import FusedLayerNorm
except ImportError:
    FusedLayerNorm = LayerNorm
    print("Please 'pip install apex'")


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    drop_path_rate: Optional[float] = None  # drop path rate
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    eva_model_name: str = None  # a valid eva model name overrides layers, width, patch_size
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16  # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    masked_language_modeling: bool = False
    fusedLN: bool = False
    xattn: bool = False
    attn_mask: bool = True


class EVA_CLIP(nn.Module):
    def __init__(
            self,
            config,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config = config
        self.eda = EDA()
        self.use_gather = config.model.use_gather
        self.eps = config.experiment.ritc_eps

        if config.experiment.ss:
            structure = config.experiment.simclr_mlp
            self.simclr_mlp = self._build_mlp(*structure)

    def _build_mlp(self, in_dim=512, mlp_dim=512, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim)
        )
    
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features


    def forward(self, input, alpha):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input['caption']

        # random deletion
        cap_new = []
        for text in texts:
            eda_alpha = self.config.experiment.eda_alpha
            cap_new.append(self.eda.random_deletion(text, eda_alpha))
        texts = cap_new

        # MLM
        if self.config.experiment.mlm:
            text_tokens, mlm_labels = tokenize(texts, context_length=self.config.experiment.text_length,
                                               mask_type='MLM')
            text_tokens = text_tokens.to(self.config.device)
            mlm_labels = mlm_labels.to(self.config.device)
        else:
            text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        # image_features, image_seq_embeddings = self.encode_image(images, return_dense=True)
        # text_features, text_seq_embeddings = self.encode_text(text_tokens, return_dense=True)
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        image_features_norm = F.normalize(image_features)
        text_features_norm = F.normalize(text_features)
        image_features_norm_gathered = self.all_gather(image_features_norm)
        text_features_norm_gathered = self.all_gather(text_features_norm)


        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        idx = ids.view(-1, 1)
        gathered_ids = self.all_gather(ids)
        idx_all = gathered_ids.view(1, -1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_i2t_targets = sim_t2i_targets = sim_targets
        if self.config.experiment.soft_label:
            with torch.no_grad():
                image_features_s = self.encode_image(images).detach()
                text_features_s = self.encode_text(text_tokens).detach()
                image_features_s_norm = F.normalize(image_features_s)
                text_features_s_norm = F.normalize(text_features_s)
                image_features_s_norm_gathered = self.all_gather(image_features_s_norm)
                text_features_s_norm_gathered = self.all_gather(text_features_s_norm)

                sim_i2t_s = logit_scale * image_features_s_norm @ text_features_s_norm_gathered.t()
                sim_t2i_s = logit_scale * text_features_s_norm @ image_features_s_norm_gathered.t()


                if self.config.model.softlabel_type == 'softmax':
                    sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
                else:
                    sim_i2t_targets = alpha * torch.sigmoid(sim_i2t_s) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * torch.sigmoid(sim_t2i_s) + (1 - alpha) * sim_targets  # soft + hard

                sim_i2t_targets = sim_i2t_targets / sim_i2t_targets.sum(1, keepdim=True)
                sim_t2i_targets = sim_t2i_targets / sim_t2i_targets.sum(1, keepdim=True)   # add
        if self.config.experiment.nitc:
            sim_i2t = logit_scale * image_features_norm @ text_features_norm_gathered.t()
            sim_t2i = logit_scale * text_features_norm @ image_features_norm_gathered.t()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            nitc_loss = (loss_i2t + loss_t2i) / 2
            ret['nitc_loss'] = nitc_loss * self.config.experiment.nitc_ratio


        if self.config.experiment.ritc:
            logits_per_image_1 = logit_scale * image_features_norm @ text_features_norm_gathered.t()
            logits_per_text_1 = logit_scale * text_features_norm @ image_features_norm_gathered.t()
            img_log = F.log_softmax(logits_per_image_1, dim=1)
            txt_log = F.log_softmax(logits_per_text_1, dim=1)

            sim_i2t_targets = sim_i2t_targets + self.eps
            sim_t2i_targets = sim_t2i_targets + self.eps
            sim_i2t_targets = sim_i2t_targets.log()
            sim_t2i_targets = sim_t2i_targets.log()
            kl_img = F.kl_div(sim_i2t_targets, img_log, log_target=True, reduction='batchmean')
            kl_txt = F.kl_div(sim_t2i_targets, txt_log, log_target=True, reduction='batchmean')
            ritc_loss = 0.5 * (kl_img + kl_txt)
            ret['ritc_loss'] = ritc_loss * self.config.experiment.ritc_ratio


        return ret
    
    
    def all_gather(self, input):
        if not self.use_gather or not is_using_distributed():
            return input
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output
    
    def calc_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                        image_features_gathered, text_features_gathered, image_features_s_gathered,
                        text_features_s_gathered,
                        sim_targets, alpha, logit_scale):
        with torch.no_grad():
            sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
            sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
            sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita,sim_i2t_s,sim_t2i_s
    
def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNorm

    visual = EVAVisionTransformer(
        img_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        num_classes=embed_dim,
        use_mean_pooling=vision_cfg.global_average_pool,  # False
        init_values=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        embed_dim=vision_cfg.width,
        depth=vision_cfg.layers,
        num_heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        qkv_bias=vision_cfg.qkv_bias,
        drop_path_rate=vision_cfg.drop_path_rate,
        norm_layer=partial(FusedLayerNorm, eps=1e-6) if vision_cfg.fusedLN else partial(norm_layer, eps=1e-6),
        xattn=vision_cfg.xattn,
        rope=vision_cfg.rope,
        postnorm=vision_cfg.postnorm,
        pt_hw_seq_len=vision_cfg.pt_hw_seq_len,  # 224/14
        intp_freq=vision_cfg.intp_freq,
        naiveswiglu=vision_cfg.naiveswiglu,
        subln=vision_cfg.subln
    )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=FusedLayerNorm if text_cfg.fusedLN else norm_layer,
        xattn=text_cfg.xattn,
        attn_mask=text_cfg.attn_mask,
    )
    return text
