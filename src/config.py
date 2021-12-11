import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torchvision import transforms

DATA_PATH = "./data"
RESULTS_DIR = "./results"
