# %%
import argparse
import copy
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from os.path import join as pjoin
from typing import Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    GuidedBackpropReLUModel,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from scipy import ndimage
from torch import nn
from torch.nn import Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
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
        weights = attention_probs
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


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
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

        self.transformer = Transformer(config)
        self.head = Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits


class Base(object):
    def __post_init__(self):
        pass


@dataclass
class DataloaderBaseConfig(Base):
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    data_dir: str = "/home/hensel/data/"
    weights_dir: str = "/home/hensel/weights"
    results_dir: str = "/home/hensel/v4"
    models_dir: str = field(init=False)
    train_dir: str = field(init=False)
    test_dir: str = field(init=False)
    cm_csv_dir: str = field(init=False)
    cm_acc_csv_dir: str = field(init=False)
    attentions_plots_dir: str = field(init=False)
    cm_plots_dir: str = field(init=False)
    cm_acc_plots_dir: str = field(init=False)
    accuracy_plots_dir: str = field(init=False)
    loss_plots_dir: str = field(init=False)
    train = False
    eval = False
    test = True
    transforms: Any = transforms.ToTensor()

    def __post_init__(self):
        super().__post_init__()
        # self.models_dir: str = pjoin(self.results_dir, "models")
        self.models_dir: str = "/home/hensel/results/models"
        self.train_dir: str = pjoin(self.results_dir, "csv/train")
        self.test_dir: str = pjoin(self.results_dir, "csv/test")
        self.cm_csv_dir: str = pjoin(self.results_dir, "csv/confusion_matrix")
        self.cm_acc_csv_dir: str = pjoin(self.results_dir, "csv/confusion_matrix_acc")
        self.attentions_plots_dir: str = pjoin(self.results_dir, "plots/attentions")
        self.cm_plots_dir: str = pjoin(self.results_dir, "plots/confusion_matrix")
        self.cm_acc_plots_dir: str = pjoin(
            self.results_dir, "plots/confusion_matrix_acc"
        )
        self.accuracy_plots_dir: str = pjoin(self.results_dir, "plots/training")
        self.loss_plots_dir: str = pjoin(self.results_dir, "plots/loss")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.cm_csv_dir, exist_ok=True)
        os.makedirs(self.cm_acc_csv_dir, exist_ok=True)
        os.makedirs(self.attentions_plots_dir, exist_ok=True)
        os.makedirs(self.cm_plots_dir, exist_ok=True)
        os.makedirs(self.cm_acc_plots_dir, exist_ok=True)
        os.makedirs(self.accuracy_plots_dir, exist_ok=True)
        os.makedirs(self.loss_plots_dir, exist_ok=True)


@dataclass
class DataloaderAug(DataloaderBaseConfig):
    train_dir: str = field(init=False)
    val_dir: str = field(init=False)
    test_dir: str = field(init=False)
    transforms: Any = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.train_dir = pjoin(self.data_dir, "mfn_224_augment_split_mini/train")
        self.val_dir = pjoin(self.data_dir, "mfn_224_augment_split_mini/val")
        self.test_dir = pjoin(self.data_dir, "mfn_224_augment_split_mini/test")


@dataclass
class DataloaderNonAug(DataloaderBaseConfig):
    train_dir: str = field(init=False)
    val_dir: str = field(init=False)
    test_dir: str = field(init=False)
    transforms: Any = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.train_dir = pjoin(self.data_dir, "mfn_224_split_mini/train")
        self.val_dir = pjoin(self.data_dir, "mfn_224_split_mini/val")
        self.test_dir = pjoin(self.data_dir, "mfn_224_split_mini/test")


@dataclass
class VitBaseConfig(Base):
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.1
    activation: Any = torch.nn.functional.gelu
    img_size: int = 224
    in_channels: int = 3
    num_classes: int = 4
    learning_rate: float = 3e-2
    weight_decay: int = 0
    momentum: float = 0.9
    num_steps: int = 500
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_accumulation_steps: int = 1
    num_epochs: int = 2
    early_stop_threshold: int = 0.001
    early_stop_patience: int = 3

    def __post_init__(self):
        super().__post_init__()


@dataclass
class VitBase(VitBaseConfig):
    pretrained_dir: str = field(init=False)
    patches: int = (16, 16)
    layers: int = 12
    hidden_size: int = 768
    mlp_size: int = 3072
    heads: int = 12

    def __post_init__(self):
        super().__post_init__()
        self.pretrained_dir = pjoin(self.weights_dir, "ViT-B_16.npz")


@dataclass
class VitLarge(VitBaseConfig):
    pretrained_dir: str = field(init=False)
    patches: int = (16, 16)
    layers: int = 24
    hidden_size: int = 1024
    mlp_size: int = 4096
    heads: int = 16

    def __post_init__(self):
        super().__post_init__()
        self.pretrained_dir = pjoin(self.weights_dir, "ViT-L_16.npz")


@dataclass
class VitHuge(VitBaseConfig):
    pretrained_dir: str = field(init=False)
    patches = (14, 14)
    layers = 32
    hidden_size = 1280
    mlp_size = 5120
    heads = 16

    def __post_init__(self):
        super().__post_init__()
        self.pretrained_dir = pjoin(self.weights_dir, "ViT-H_14.npz")


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


# class GradCAM(BaseCAM):
#     def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
#         super(GradCAM, self).__init__(model, target_layers, use_cuda, reshape_transform)

#     def get_cam_weights(
#         self, input_tensor, target_layer, target_category, activations, grads
#     ):
#         # grads = np.maximum(grads, 0)
#         return np.max(grads, axis=(2, 3))


# def predict(image_path):
#     model_path = "/home/hensel/projects/vit-masked-face-net/models/vit_huge_14_augment_pretrained_best_model.pth"
#     classes_file = "/home/hensel/projects/vit-masked-face-net/classes.txt"

#     config = VitHugeAugPretrained()
#     model = VisionTransformer(config)
#     checkpoint = torch.load(model_path, map_location="cpu")
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     transform = transforms.Compose(
#         [
#             transforms.Resize(224),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )

#     img = Image.open(image_path)
#     batch_t = torch.unsqueeze(transform(img), 0)

#     # resnet.eval()
#     out, attn_weights = model(batch_t)

#     with open(classes_file) as f:
#         classes = [line.strip() for line in f.readlines()]

#     prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     _, indices = torch.sort(out, descending=True)
#     return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


# st.set_option("deprecation.showfileUploaderEncoding", False)

# # st.title(
# #     "Demo Klasifikasi Cara Penggunaan Masker Menggunakan Arsitektur Vision Transformer Dengan Metode Transfer Learning Dan Augmentasi Data"
# # )
# st.write("")
# st.header(
#     "Demo Klasifikasi Cara Penggunaan Masker Menggunakan Arsitektur Vision Transformer Dengan Metode Transfer Learning Dan Augmentasi Data"
# )
# st.write("Hensel Donato Jahja - 185150200111064")
# file_up = st.file_uploader("Upload an image", type="jpg")

# if file_up is not None:
#     image = Image.open(file_up)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)
#     st.write("")
#     st.write("Menunggu prediksi...")
#     labels = predict(file_up)
#     print(labels)

#     for i in labels:
#         class_name = i[0].split(",")
#         class_name = class_name[1].split("_")
#         class_name = " ".join(class_name)

#         st.write("Prediksi : ", class_name, ",  dengan score: ", i[1])


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# target_layers = [model.blocks[-1].norm1]

# cam = GradCAM(
#     model=model,
#     target_layers=target_layers,
#     use_cuda=args.use_cuda,
#     reshape_transform=reshape_transform,
# )

# rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
# rgb_img = cv2.resize(rgb_img, (224, 224))
# rgb_img = np.float32(rgb_img) / 255
# input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# target_category = None
# cam.batch_size = 32

# grayscale_cam = cam(
#     input_tensor=input_tensor,
#     target_category=target_category,
#     eigen_smooth=True,
#     aug_smooth=True,
# )

# # Here grayscale_cam has only one image in the batch
# grayscale_cam = grayscale_cam[0, :]

# cam_image = show_cam_on_image(rgb_img, grayscale_cam)
# # cv2.imwrite(f"{args.method}_cam.jpg", cam_image)

from pytorch_grad_cam.ablation_layer import AblationLayerVit


def grad_cam(model, image):

    if "Vision" in model.__class__.__name__:
        target_layers = [model.blocks[-1].norm1]
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=reshape_transform,
            # ablation_layer=AblationLayerVit,
        )
    else:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None,
        )
    rgb_img = image.convert("RGB")
    rgb_img = np.array(rgb_img)
    rgb_img = rgb_img[:, :, ::-1].copy()
    # rgb_img = cv2.imread(image, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    target_category = None
    cam.batch_size = 32

    grayscale_cam = cam(
        input_tensor=input_tensor,
        target_category=target_category,
        eigen_smooth=True,
        aug_smooth=True,
    )

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return cam_image
    # cv2.imwrite(f"{args.method}_cam.jpg", cam_image)


st.set_page_config(layout="wide")
vit_model_path = "/home/hensel/projects/vit-masked-face-net/models/vit_huge_14_augment_pretrained_best_model.pth"
vit_checkpoint = torch.load(
    "/home/hensel/projects/vit-masked-face-net/diet_11.pt",
    map_location=torch.device("cpu"),
)
vit = torch.hub.load(
    "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=False
)
vit.head = nn.Linear(192, 4)
vit.load_state_dict(vit_checkpoint)

resnet_152 = models.resnet152(pretrained=False)
resnet_152.fc = nn.Linear(2048, 4)
resnet_checkpoint = torch.load(
    "/home/hensel/projects/vit-masked-face-net/models/resnet152_aug_pretrained_best_model.pth",
    map_location=torch.device("cpu"),
)

resnet_152.load_state_dict(resnet_checkpoint["model_state_dict"])


def predict(model, image_path):
    classes_file = "/home/hensel/projects/vit-masked-face-net/classes.txt"

    # config = VitHugeAugPretrained()
    # model = VisionTransformer(config)
    # checkpoint = torch.load(model_path, map_location="cpu")
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    out = model(batch_t)
    torch.cuda.empty_cache()

    with open(classes_file) as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


st.set_option("deprecation.showfileUploaderEncoding", False)

# st.title(
#     "Demo Klasifikasi Cara Penggunaan Masker Menggunakan Arsitektur Vision Transformer Dengan Metode Transfer Learning Dan Augmentasi Data"
# )
st.write("")
st.header(
    "Demo Klasifikasi Cara Penggunaan Masker Menggunakan Arsitektur Vision Transformer Dengan Metode Transfer Learning Dan Augmentasi Data"
)
st.write("Hensel Donato Jahja - 185150200111064")
col1, col2, col3 = st.columns(3)
with col1:
    file_up = st.file_uploader("Upload an image", type="jpg")
    # st.header("Image ")
    # st.image(image, caption=file_up.name, use_column_width=True)
if file_up is not None:

    # print(file_up.name)
    image = Image.open(file_up)
    with col1:
        st.header(f"Image {file_up.name}")
        st.image(image, caption=file_up.name, width=500)
    # st.write("")
    # st.write("Menunggu prediksi...")
    # col1, col2 = st.columns(2)
    with col2:
        st.header("Hasil Prediksi Vision Transformer")
        labels = predict(vit, file_up)
        for i in labels:
            class_name = i[0].split(",")
            class_name = class_name[1].split("_")
            class_name = " ".join(class_name)

            st.write("Prediksi : ", class_name, ",  dengan score: ", i[1])
        cam = grad_cam(vit, image)
        st.image(cam, caption="GradCam Vision Transformers", width=500)
    with col3:
        st.header("Hasil Prediksi ResNet 152")
        labels = predict(resnet_152, file_up)
        for i in labels:
            class_name = i[0].split(",")
            class_name = class_name[1].split("_")
            class_name = " ".join(class_name)

            st.write("Prediksi : ", class_name, ",  dengan score: ", i[1])
        cam = grad_cam(resnet_152, image)
        st.image(cam, caption="GradCam ResNet 152", width=500)
    # labels = predict(file_up)
    # print(labels)
    # st.write("")
    # st.write("Hasil GradCam")

    # for i in labels:
    #     class_name = i[0].split(",")
    #     class_name = class_name[1].split("_")
    #     class_name = " ".join(class_name)

    #     st.write("Prediksi : ", class_name, ",  dengan score: ", i[1])
    # cam = grad_cam(image)
    # st.image(cam, caption="GradCam", width=400)
