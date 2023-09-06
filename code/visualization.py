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
import matplotlib.pyplot as plt
import random

from labels import labels

def visualize_samples(gt_masks, logits_masks, save_dir, to_sample=True, num_samples=10):
    # create directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logits_tensor = torch.from_numpy(logits_masks)
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=gt_masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    print(logits_tensor.shape)
    pred_labels = logits_tensor.detach().cpu().numpy()
    # sample images if needed
    if to_sample:
        indices = np.random.randint(0, len(gt_masks), num_samples)
        gt_masks = gt_masks[indices]
        pred_labels = pred_labels[indices]

    for i, (pred_mask, gt_mask) in enumerate(zip(pred_labels, gt_masks)):  
        visualize_sample(pred_mask, gt_mask, save_dir, i)

def visualize_sample(prediction_mask, gt_mask, save_dir, index):
    pred_seg = np.zeros((prediction_mask.shape[0], prediction_mask.shape[1], 3), dtype=np.uint8)
    gt_seg = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    for label in labels:
        pred_seg[prediction_mask == label.id] = label.color
        gt_seg[gt_mask == label.id] = label.color
    # save images
    plot_image(pred_seg, f"Prediction {index}", os.path.join(save_dir, f"prediction_{index}.png"))
    plot_image(gt_seg, f"Ground Truth {index}", os.path.join(save_dir, f"ground_truth_{index}.png"))

def plot_image(image, title, save_path):
    # Plot only segmentation map
    plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.axis("off")
    plt.imshow(image)
    # save image
    plt.savefig(save_path)

        