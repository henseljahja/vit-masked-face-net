import argparse
import copy
import os
import random
import shutil
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, vgg19
from tqdm import tqdm
