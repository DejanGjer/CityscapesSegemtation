from datasets import load_dataset
from labels import labels, Label
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter, ToTensor
import numpy as np
import evaluate
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
import wandb
import os



