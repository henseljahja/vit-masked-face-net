import copy
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from os.path import join as pjoin
from typing import Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import torch
import torch.distributed as dist
import torch.nn as nn
from matplotlib.ticker import AutoMinorLocator
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.nn import Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# %%
indonesia_timezone = pytz.timezone("Asia/Jakarta")
now = datetime.now(indonesia_timezone)
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
RESULTS_DIR = f"/home/hensel/output_v10/output_v10_{dt_string}"
os.makedirs(RESULTS_DIR, exist_ok=True)
test_dir: str = "/home/hensel/results"

# %%
@dataclass
class DataloaderBaseConfig:
    seed: int = 42
    batch_size: int = 16
    train_batch_size: int = 16
    test_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    # train_dir: str = "./data_split/val/"
    # val_dir: str = "./data_split/val/"
    # test_dir: str = "./data_split/test/"
    # train_dir: str = "/home/hensel/data/train/"
    # val_dir: str = "/home/hensel/data/val/"
    # test_dir: str = "/home/hensel/data/test/"
    # train_dir: str = "/home/hensel/test_data/train/"
    # val_dir: str = "/home/hensel/test_data/val/"
    # test_dir: str = "/home/hensel/test_data/test/"
    # train_dir: str = "/home/hensel/data_224/train/"
    # val_dir: str = "/home/hensel/data_224/val/"
    # test_dir: str = "/home/hensel/data_224/test/"
    plots_dir: str = f"{RESULTS_DIR}/plots/"
    results_dir: str = f"{RESULTS_DIR}/results/"
    models_dir: str = f"{RESULTS_DIR}/models"
    attentions_dir: str = f"{RESULTS_DIR}/attentions/"
    model_test_dir: str = f"{test_dir}/models/"
    plot_test_dir: str = f"{test_dir}/test/plots"
    results_test_dir: str = f"{test_dir}/test/results"
    transforms: Any = transforms.ToTensor()

    def __post_init__(self):
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.attentions_dir, exist_ok=True)
        os.makedirs(self.model_test_dir, exist_ok=True)
        os.makedirs(self.plot_test_dir, exist_ok=True)
        os.makedirs(self.results_test_dir, exist_ok=True)


#     output_dir: str = "./"


@dataclass
class DataloaderAug(DataloaderBaseConfig):
    train_dir: str = "/home/hensel/mfn/mfn_224_augment_split_mini/train/"
    val_dir: str = "/home/hensel/mfn/mfn_224_augment_split_mini/val/"
    test_dir: str = "/home/hensel/mfn/mfn_224_augment_split_mini/test/"
    # transforms: Any = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         #             transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    #         transforms.GaussianBlur(kernel_size=51),
    #         transforms.RandomVerticalFlip(p=0.5),
    #         transforms.ColorJitter(brightness=0.5, hue=0.3),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ]
    # )


@dataclass
class DataloaderNonAug(DataloaderBaseConfig):
    train_dir: str = "/home/hensel/mfn/mfn_224_split_mini/train/"
    val_dir: str = "/home/hensel/mfn/mfn_224_split_mini/val/"
    test_dir: str = "/home/hensel/mfn/mfn_224_split_mini/test/"
    # transforms: Any = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ]
    # )


@dataclass
class VitBaseConfig:
    #     patches:int = 16
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.1
    classifier: str = "token"
    representasion_size: Any = None
    activation: Any = torch.nn.functional.gelu
    img_size: int = 224
    in_channels: int = 3
    vis: bool = True
    num_classes: int = 4
    # num_classes: int = 30
    zero_head: bool = True
    dataset: str = "MaskedFaceNet"
    # eval_every: int = 6
    learning_rate: float = 3e-2
    weight_decay: int = 0
    momentum: float = 0.9
    num_steps: int = 500
    decay_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    local_rank: int = -1
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: int = 0
    fp16_opt_level: str = "O2"
    loss_scale: int = 0
    num_epochs: int = 20
    early_stop_threshold: int = 0.001
    early_stop_patience: int = 3
    # early_stop_improvement_threshold: int = 0.05
    # early_stop_divergence_threshold: int = 0.1


@dataclass
class VitBase(VitBaseConfig):
    pretrained_dir: str = "/home/hensel/weights/ViT-B_16.npz"
    patches: int = (16, 16)
    layers: int = 12
    hidden_size: int = 768
    mlp_size: int = 3072
    heads: int = 12


@dataclass
class VitLarge(VitBaseConfig):
    pretrained_dir: str = "/home/hensel/weights/ViT-L_16.npz"
    patches: int = (16, 16)
    layers: int = 24
    hidden_size: int = 1024
    mlp_size: int = 4096
    heads: int = 16


@dataclass
class VitHuge(VitBaseConfig):
    pretrained_dir: str = "/home/hensel/weights/ViT-H_14.npz"
    patches = (14, 14)
    layers = 32
    hidden_size = 1280
    mlp_size = 5120
    heads = 16


""" ViT Base"""


@dataclass
class VitBaseAugPretrained(VitBase, DataloaderAug):
    name: str = "vit_base_16_augment_pretrained"
    pretrained: bool = True


@dataclass
class VitBasePretrained(VitBase, DataloaderNonAug):
    name: str = "vit_base_16_pretrained"
    pretrained: bool = True


@dataclass
class VitBaseAug(VitBase, DataloaderAug):
    name: str = "vit_base_16_augment"
    pretrained: bool = False


@dataclass
class VitBase(VitBase, DataloaderNonAug):
    name: str = "vit_base_16"
    pretrained: bool = False


""" ViT Large"""


@dataclass
class VitLargeAugPretrained(VitLarge, DataloaderAug):
    name: str = "vit_large_16_augment_pretrained"
    pretrained: bool = True


@dataclass
class VitLargePretrained(VitLarge, DataloaderNonAug):
    name: str = "vit_large_16_pretrained"
    pretrained: bool = True


@dataclass
class VitLargeAug(VitLarge, DataloaderAug):
    name: str = "vit_large_16_augment"
    pretrained: bool = False


@dataclass
class VitLarge(VitLarge, DataloaderNonAug):
    name: str = "vit_large_16"
    pretrained: bool = False


""" ViT Huge"""


@dataclass
class VitHugeAugPretrained(VitHuge, DataloaderAug):
    name: str = "vit_huge_14_augment_pretrained"
    pretrained: bool = True


@dataclass
class VitHugePretrained(VitHuge, DataloaderNonAug):
    name: str = "vit_huge_14_pretrained"
    pretrained: bool = True


@dataclass
class VitHugeAug(VitHuge, DataloaderAug):
    name: str = "vit_huge_14_augment"
    pretrained: bool = False


@dataclass
class VitHuge(VitHuge, DataloaderNonAug):
    name: str = "vit_huge_14"
    pretrained: bool = False


#%%
# # Resnet


@dataclass
class ResnetBaseConfig:
    img_size: int = 224
    num_classes: int = 4
    learning_rate: float = 3e-2
    weight_decay: int = 0
    momentum: float = 0.9
    seed: int = 42
    num_epochs: int = 20
    early_stop_threshold: int = 0.001
    early_stop_patience: int = 3
    num_steps: int = 500
    warmup_steps: int = 100
    max_grad_norm: float = 1.0


@dataclass
class Resnet152(ResnetBaseConfig, DataloaderNonAug):
    name = "resnet152"
    pretrained = False


@dataclass
class Resnet152Aug(ResnetBaseConfig, DataloaderAug):
    name = "resnet152_aug"
    pretrained = False


@dataclass
class Resnet152AugPretrained(ResnetBaseConfig, DataloaderAug):
    name = "resnet152_aug_pretrained"
    pretrained = True


@dataclass
class Resnet152Pretrained(ResnetBaseConfig, DataloaderNonAug):
    name = "resnet152_pretrained"
    pretrained = True


@dataclass
class Resnet50(ResnetBaseConfig, DataloaderNonAug):
    name = "resnet50"
    pretrained = False


@dataclass
class Resnet50Aug(ResnetBaseConfig, DataloaderAug):
    name = "resnet50_aug"
    pretrained = False


@dataclass
class Resnet50AugPretrained(ResnetBaseConfig, DataloaderAug):
    name = "resnet50_aug_pretrained"
    pretrained = True


@dataclass
class Resnet50Pretrained(ResnetBaseConfig, DataloaderNonAug):
    name = "resnet50_pretrained"
    pretrained = True


# %% [markdown]
# ## Resnet


# %%
# from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights):
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.vis = config.vis
        self.num_attention_heads = config.heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.attention_dropout_rate)
        self.proj_dropout = Dropout(config.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_size)
        self.fc2 = Linear(config.mlp_size, config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.attention_dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(config.img_size)

        patch_size = _pair(config.patches)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        self.patch_embeddings = Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches + 1, config.hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.attention_dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")])
            )
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")])
            )
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vis = config.vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.num_classes = config.num_classes
        self.zero_head = config.zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config)
        self.head = Linear(config.hidden_size, config.num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"].transpose([3, 2, 0, 1]))
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info(
                    "load_pretrained: resized variant: %s to %s"
                    % (posemb.size(), posemb_new.size())
                )
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


# %%
# logger = logging.getLogger(__name__)


class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)),
        )


# %%
# logger = logging.getLogger(__name__)


def get_loader(
    config: dataclass,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    trainset = ImageFolder(root=config.train_dir, transform=config.transforms)
    valset = ImageFolder(root=config.val_dir, transform=config.transforms)
    testset = ImageFolder(root=config.test_dir, transform=config.transforms)

    trainset_sampler = RandomSampler(trainset)
    valset_sampler = SequentialSampler(valset)
    testset_sampler = SequentialSampler(testset)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=trainset_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=valset_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=testset_sampler,
    )

    return train_loader, val_loader, test_loader


# %%
# logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self, config):
        self.config = config
        self.metrics = {
            "epoch": [],
            "train_time": [],
            "train_accuracy": [],
            "train_loss": [],
            "eval_time": [],
            "eval_accuracy": [],
            "eval_loss": [],
        }
        self.epoch = 0

        # self.early_stop_counter = 0
        # self.eval_best_loss = float("inf")
        # self.early_stop_patience = self.config.early_stop_patience
        # self.early_stop_threshold: int = self.config.early_stop_threshold
        # self.early_stop_patience: int = self.config.early_stop_patience

    def reset(self):
        self.train_time = 0
        self.train_accuracy = 0
        self.train_loss = 0
        self.eval_time = 0
        self.eval_accuracy = 0
        self.eval_loss = 0

    def train_update(self, train_time, accuracy, loss):
        self.reset()
        self.train_time = train_time
        self.metrics["train_time"].append(self.train_time)

        self.train_accuracy = accuracy
        self.train_loss = loss

        self.metrics["train_accuracy"].append(accuracy)
        self.metrics["train_loss"].append(loss)

    def eval_update(self, eval_time, accuracy, loss):
        self.reset()

        self.epoch += 1
        self.metrics["epoch"].append(self.epoch)

        self.eval_time = eval_time
        self.metrics["eval_time"].append(self.eval_time)

        self.eval_accuracy = accuracy
        self.eval_loss = loss

        self.metrics["eval_accuracy"].append(accuracy)
        self.metrics["eval_loss"].append(loss)

    def to_pandas(self):
        metrics = self.metrics
        df = pd.DataFrame(metrics)
        df = df.add_prefix(f"{self.config.name}_")
        return df

    def to_csv(self):
        self.make_dirs()
        df = self.to_pandas()
        df.to_csv(f"{self.config.results_dir}/{self.config.name}.csv", index=False)

    def to_plot(self):
        df = self.to_pandas()
        title = f"{self.config.name}"
        title = title.split("_")
        title = " ".join(title).title()

        ax1 = df[[x for x in df.columns if "loss" not in x and "time" not in x]].plot(
            x=f"{self.config.name}_epoch",
            figsize=(15, 10),
            linewidth=5,
            kind="line",
            legend=True,
            fontsize=16,
        )
        ax1.legend(
            loc="lower right",
            prop={"size": 20},
        )
        ax1.grid(True)
        ax1.set_title(title + " Accuracy", fontsize=30)
        ax1.set_xlabel("Epoch", fontsize=20)
        ax1.set_ylabel("Accuracy", fontsize=20)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.grid(which="major", color="#CCCCCC", linestyle="--")
        ax1.grid(which="minor", color="#CCCCCC", linestyle=":")
        ax1.figure.savefig(
            self.config.plots_dir + "/" + self.config.name + "_accuracy.png"
        )

        ax2 = df[
            [x for x in df.columns if "accuracy" not in x and "time" not in x]
        ].plot(
            x=f"{self.config.name}_epoch",
            figsize=(15, 10),
            linewidth=5,
            kind="line",
            legend=True,
            fontsize=16,
        )
        ax2.legend(
            loc="lower right",
            prop={"size": 20},
        )
        ax2.grid(True)
        ax2.set_xlabel("Epoch", fontsize=20)
        ax2.set_ylabel("Loss", fontsize=20)
        ax2.set_title(title + " Loss", fontsize=30)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.grid(which="major", color="#CCCCCC", linestyle="--")
        ax2.grid(which="minor", color="#CCCCCC", linestyle=":")
        ax2.figure.savefig(self.config.plots_dir + "/" + self.config.name + "_loss.png")
        plt.close("all")

    def get_metrics(self):
        return self.metrics

    def make_dirs(self):
        os.makedirs(self.config.plots_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        os.makedirs(self.config.models_dir, exist_ok=True)

    def early_stop(self, model):
        if self.eval_best_loss - self.eval_loss > self.early_stop_threshold:
            best_model = copy.deepcopy(model)
            os.makedirs(self.config.models_dir, exist_ok=True)
            model_dir = (
                f"{self.config.models_dir}/{self.config.name}_Early_Stopped_Model.pt"
            )
            if os.path.exists(model_dir):
                os.remove(model_dir)

            torch.save(best_model.state_dict(), model_dir)
            self.eval_best_loss = self.eval_loss
            self.early_stop_counter = 0

        elif self.eval_best_loss - self.eval_loss < self.early_stop_threshold:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.early_stop_patience:
                return True
        # eval_loss_diff = self.metrics["eval_loss"][-1] - self.metrics["eval_loss"][-2]
        # if eval_loss_diff > self.config.early_stop_improvement_threshold:
        # #     self.early_stop_counter = 0
        # if (eval_loss_diff > -(self.config.early_stop_threshold)) or abs(eval_loss_diff) < abs(self.config.early_stop_threshold):
        #     self.early_stop_counter += 1
        # if abs(eval_loss_diff) < abs(self.early_stop_threshold):
        #     self.early_stop_counter += 1
        # if self.eval_loss < self.early_stop_threshold * -1:
        #     self.early_stop_counter = 0
        # if eval_loss_diff < -(self.config.early_stop_threshold):
        #     self.early_stop_counter += 1
        # if self.eval_loss > self.early_stop_threshold:
        #     self.early_stop_counter += 1
        #     if self.early_stop_counter >= self.early_stop_patience:
        #         return True
        # if self.metrics["eval_loss"][-1] > self.config.early_stop_divergence_threshold:
        #     return True


def train(config, model):
    train_loader, eval_loader, test_loader = get_loader(config)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    t_total = config.num_steps

    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=config.warmup_steps, t_total=t_total
    )

    metrics = Metrics(config)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    global_step, best_acc = 0.0, 0.0
    # model, optimizer = amp.initialize(
    #     models=model, optimizers=optimizer, opt_level=config.fp16_opt_level
    # )
    # amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
    # epoch = 0
    for epoch in range(config.num_epochs):
        # while True:
        # epoch += 1
        model.train()
        train_iterator = tqdm(
            train_loader,
            desc=config.name
            + " Training Epoch X / X : Batch X / X) (Acc = X, Loss = X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
        )
        loss_fct = CrossEntropyLoss()

        train_running_loss = 0.0
        train_running_corrects = 0.0

        train_epoch_acc = 0.0
        train_epoch_loss = 0.0

        eval_running_loss = 0.0
        eval_running_corrects = 0.0

        eval_epoch_loss = 0.0
        eval_epoch_acc = 0.0

        train_start_time = time.process_time()

        for batch, (inputs, labels) in enumerate(train_iterator):

            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()

            if "vit" in config.name:
                logits, attn_weights = model(inputs)
            else:
                logits = model(inputs)
            loss = loss_fct(logits.view(-1, config.num_classes), labels.view(-1))

            _, preds = torch.max(logits, 1)
            train_running_acc = (preds == labels).sum() / len(labels)

            train_running_loss += loss.item() * inputs.size(0)
            train_running_corrects += torch.sum(preds == labels.data)

            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            scheduler.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(
            #     amp.master_params(optimizer), config.max_grad_norm
            # )
            global_step += 1

            batch_cnt = batch + 1
            epoch_cnt = epoch + 1

            train_iterator.set_description(
                config.name
                + " Training Epoch %d : Batch %d / %d) (Acc = %2.5f, Loss = %2.5f)"
                % (
                    epoch_cnt,
                    # config.num_epochs,
                    batch_cnt,
                    len(train_loader),
                    train_running_acc.item(),
                    loss.item(),
                )
            )
        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        train_epoch_acc = (train_running_corrects / len(train_loader.dataset)).item()

        train_end_time = time.process_time() - train_start_time
        metrics.train_update(train_end_time, train_epoch_acc, train_epoch_loss)
        # print(f"Metrics {metrics.get_metrics()}")

        model.eval()
        eval_iterator = tqdm(
            eval_loader,
            desc=config.name + " Eval Epoch X / X : Batch X / X) (Acc = X, Loss = X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
        )
        eval_start_time = time.process_time()
        for batch, (inputs, labels) in enumerate(eval_iterator):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            with torch.no_grad():
                if "vit" in config.name:
                    logits, atttn_weights = model(inputs)
                else:
                    logits = model(inputs)
                loss = loss_fct(logits, labels)
                _, preds = torch.max(logits, 1)

            eval_running_acc = ((preds == labels).sum() / len(labels)).item()

            eval_running_loss += loss.item() * inputs.size(0)
            eval_running_corrects += torch.sum(preds == labels.data)

            batch_cnt = batch + 1
            epoch_cnt = epoch + 1

            eval_iterator.set_description(
                config.name
                + " Eval Epoch %d : Batch %d / %d) (Acc = %2.5f, Loss = %2.5f)"
                % (
                    epoch_cnt,
                    # config.num_epochs,
                    batch_cnt,
                    len(eval_loader),
                    eval_running_acc,
                    loss.item(),
                )
            )

        eval_epoch_loss = eval_running_loss / len(eval_loader.dataset)
        eval_epoch_acc = (eval_running_corrects / len(eval_loader.dataset)).item()

        eval_end_time = time.process_time() - eval_start_time

        metrics.eval_update(eval_end_time, eval_epoch_acc, eval_epoch_loss)

        # print(f"Metrics {metrics.get_metrics()}")

        metrics.to_csv()
        metrics.to_plot()

        if eval_epoch_acc > best_acc:

            best_acc = eval_epoch_acc
            best_model = copy.deepcopy(model)
            os.makedirs(config.models_dir, exist_ok=True)
            model_dir = f"{config.models_dir}/{config.name}_best_model.pth"
            if os.path.exists(model_dir):
                os.remove(model_dir)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": eval_epoch_loss,
                },
                model_dir,
            )
            logger.info(f"Best accuracy at {best_acc}")
            logger.info(f"Saved model to {model_dir}")


def test(config, model):
    train_loader, eval_loader, test_loader = get_loader(config)

    metrics = {
        "test_acc": [],
        "test_loss": [],
        "test_time": [],
    }

    test_iterator = tqdm(
        test_loader,
        desc=config.name + " Test Epoch X / X : Batch X / X) (Acc = X, Loss = X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
    )
    classes = test_loader.dataset.class_to_idx

    loss_fct = CrossEntropyLoss()
    test_running_loss = 0.0
    test_running_corrects = 0.0

    test_epoch_loss = 0.0
    test_epoch_acc = 0.0

    checkpoint = torch.load(
        config.model_test_dir + "/" + config.name + "_best_model.pth"
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(config.device)
    test_start_time = time.process_time()
    for batch, (inputs, labels) in enumerate(test_iterator):
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        with torch.no_grad():
            if "vit" in config.name:
                logits, atttn_weights = model(inputs)
            else:
                logits = model(inputs)
            loss = loss_fct(logits, labels)
            _, preds = torch.max(logits, 1)

        test_running_acc = ((preds == labels).sum() / len(labels)).item()

        test_running_loss += loss.item() * inputs.size(0)
        test_running_corrects += torch.sum(preds == labels.data)

        test_iterator.set_description(
            config.name
            + " Test Batch %d / %d) (Acc = %2.5f, Loss = %2.5f)"
            % (
                batch,
                len(test_loader),
                test_running_acc,
                loss.item(),
            )
        )

    test_epoch_loss = test_running_loss / len(test_loader.dataset)
    test_epoch_acc = (test_running_corrects / len(test_loader.dataset)).item()

    test_end_time = time.process_time() - test_start_time

    metrics["test_acc"].append(test_epoch_acc)
    metrics["test_loss"].append(test_epoch_loss)
    metrics["test_time"].append(test_end_time)

    pd.DataFrame(metrics).to_csv(
        f"{config.results_test_dir}/{config.name}_test.csv", index=False
    )
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    im = Image.open("/home/hensel/results/Mask_0.png")
    x = transform(im)
    model.to(torch.device("cpu"))
    logits, att_mat = model(x.unsqueeze(0))
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original", fontsize=16)
    ax2.set_title("Attention Map", fontsize=16)

    ax1.imshow(im)
    ax2.imshow(result)

    fig.savefig(f"{config.plot_test_dir}/{config.name}_attention_v1.png")

    plt.close()

    im = Image.open("/home/hensel/results/Mask_Mouth_Chin_99999.png")
    x = transform(im)
    model.to(torch.device("cpu"))
    logits, att_mat = model(x.unsqueeze(0))
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax2.set_title("Attention Map")

    ax1.imshow(im)
    ax2.imshow(result)

    fig.savefig(f"{config.plot_test_dir}/{config.name}_attention_v2.png")

    plt.close()


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True
    return model


""" Aug Pretrained Model """


def vit_base_aug_pretrained():
    config = VitBaseAugPretrained()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_large_aug_pretrained():
    config = VitLargeAugPretrained()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_huge_aug_pretrained():
    config = VitHugeAugPretrained()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


""" Non Aug Pretrained Model """


def vit_base_pretrained():
    config = VitBasePretrained()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_large_pretrained():
    config = VitLargePretrained()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_huge_pretrained():
    config = VitHugePretrained()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


"""Aug Non Pretrained Model """


def vit_base_aug():
    config = VitBaseAug()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_large_aug():
    config = VitLargeAug()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_huge_aug():
    config = VitHugeAug()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


"""Non Aug Non Pretrained Model """


def vit_base():
    config = VitBase()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_large():
    config = VitLarge()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def vit_huge():
    config = VitHuge()
    config.device = device
    model = VisionTransformer(config)
    if config.pretrained == True:
        model.load_from(np.load(config.pretrained_dir))
        model = freeze_layers(model)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


"""Res Net 152"""


def resnet_152():
    config = Resnet152()
    config.device = device
    model = models.resnet152(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def resnet_152_aug():
    config = Resnet152Aug()
    config.device = device
    model = models.resnet152(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def resnet_152_aug_pretrained():
    config = Resnet152AugPretrained()
    config.device = device
    model = models.resnet152(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def resnet_152_pretrained():
    config = Resnet152Pretrained()
    config.device = device
    model = models.resnet152(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


"""Res Net 50"""


def resnet_50():
    config = Resnet50()
    config.device = device
    model = models.resnet50(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def resnet_50_aug():
    config = Resnet50Aug()
    config.device = device
    model = models.resnet50()(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def resnet_50_aug_pretrained():
    config = Resnet50AugPretrained()
    config.device = device
    model = models.resnet50(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


def resnet_50_pretrained():
    config = Resnet50Pretrained()
    config.device = device
    model = models.resnet50(pretrained=config.pretrained)
    if config.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048, config.num_classes)
    model.to(config.device)
    logger.info(f"Starting Training {config.name}")
    # train(config, model)
    test(config, model)
    torch.cuda.empty_cache()


#%%
if __name__ == "__main__":
    """Base Model"""
    vit_base_pretrained()
    vit_base()

    """ Base Aug Model """
    vit_base_aug_pretrained()
    vit_base_aug()

    """ Large Model """
    vit_large_pretrained()
    vit_large()

    """ Large Aug Model """
    vit_large_aug_pretrained()
    vit_large_aug()

    """ Huge Model """
    vit_huge_pretrained()
    vit_huge()

    """ Huge Aug Model """
    vit_huge_aug_pretrained()
    vit_huge_aug()

    """Resnet 152"""
    resnet_152_pretrained()
    resnet_152()

    resnet_152_aug_pretrained()
    resnet_152_aug()

    """Resnet 50"""
    resnet_50_pretrained()
    resnet_50()

    resnet_50_aug_pretrained()
    resnet_50_aug()

    logger.info("Training Finished")
