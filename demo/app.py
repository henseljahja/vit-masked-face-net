# %%

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from cv2 import COLOR_RGB2BGR, cvtColor, resize
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch import nn
from torchvision import models, transforms


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def grad_cam(model, image):

    if "Vision" in model.__class__.__name__:
        target_layers = [model.blocks[-1].norm1]
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=reshape_transform,
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
    rgb_img = resize(rgb_img, (224, 224))
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
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = cvtColor(cam_image, COLOR_RGB2BGR)

    return cam_image


# vit_model_path = "vit_huge_14_augment_pretrained_best_model.pth"
vit_checkpoint = torch.load(
    "vit_aug_pretrained_best_model.pt",
    map_location=torch.device("cpu"),
)
vit = torch.hub.load(
    "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=False
)
vit.head = nn.Linear(192, 4)
vit.load_state_dict(vit_checkpoint)

# resnet_152 = models.resnet152(pretrained=False)
# resnet_152.fc = nn.Linear(2048, 4)
# resnet_checkpoint = torch.load(
#     "resnet152_aug_pretrained_best_model.pth", map_location=torch.device("cpu"),
# )

# resnet_152.load_state_dict(resnet_checkpoint["model_state_dict"])


def predict(model, image_path):
    classes_file = "classes.txt"

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

st.write("")
st.header(
    "Demo Klasifikasi Cara Penggunaan Masker Menggunakan Arsitektur Vision Transformer Dengan Metode Transfer Learning Dan Augmentasi Data"
)
st.write("Hensel Donato Jahja - 185150200111064")
file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    # print(file_up.name)
    image = Image.open(file_up)
    st.image(image, caption=file_up.name, use_column_width=True)
    st.write("")
    st.write("Menunggu prediksi...")
    # col1, col2 = st.columns(2)
    # with col1:
    st.header("Hasil Prediksi Vision Transformer")
    labels = predict(vit, file_up)
    for i in labels:
        class_name = i[0].split(",")
        class_name = class_name[1].split("_")
        class_name = " ".join(class_name)

        st.write("Prediksi : ", class_name, ",  dengan score: ", i[1])
    cam = grad_cam(vit, image)
    st.image(cam, caption="GradCam Vision Transformers", use_column_width=True)
    # with col2:
    #     st.header("Hasil Prediksi ResNet 152")
    #     labels = predict(resnet_152, file_up)
    #     for i in labels:
    #         class_name = i[0].split(",")
    #         class_name = class_name[1].split("_")
    #         class_name = " ".join(class_name)

    #         st.write("Prediksi : ", class_name, ",  dengan score: ", i[1])
    #     cam = grad_cam(resnet_152, image)
    #     st.image(cam, caption="GradCam ResNet 152", use_column_width=True)
