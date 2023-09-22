from labels import labels
import numpy as np 
import numpy as np
import torch
from torch import nn
import os
from PIL import Image

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
    print("GT MASKS INFO")
    print(gt_masks.shape)
    print(np.unique(gt_masks[0]))
    pred_labels = logits_tensor.detach().cpu().numpy()
    # sample images if needed
    if to_sample:
        indices = np.random.randint(0, len(gt_masks), num_samples)
        gt_masks = gt_masks[indices]
        pred_labels = pred_labels[indices]

    print("GT MASKS AFTER SAMPLING")
    print(gt_masks.shape)
    for i, (pred_mask, gt_mask) in enumerate(zip(pred_labels, gt_masks)):  
        visualize_sample(pred_mask, gt_mask, save_dir, i)

def visualize_sample(prediction_mask, gt_mask, save_dir, index):
    # get all label ids from gt_mask
    print(np.unique(gt_mask))
    pred_seg = np.zeros((prediction_mask.shape[0], prediction_mask.shape[1], 3), dtype=np.uint8)
    gt_seg = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    for label in labels:
        pred_seg[prediction_mask == label.id] = label.color
        gt_seg[gt_mask == label.id] = label.color
    # save images
    save_image(pred_seg, os.path.join(save_dir, f"prediction_{index}.png"))
    save_image(gt_seg, os.path.join(save_dir, f"ground_truth_{index}.png"))

def visualize_mask(mask, save_dir, index, prefix=""):
    # create directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if mask is given in 3 channels format convert it to 1 channel
    if mask.shape[-1] == 3:
        mask = mask[:, :, 0]
    # get all label ids from gt_mask
    seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in labels:
        seg[mask == label.id] = label.color
    # save images
    save_image(seg, os.path.join(save_dir, f"{prefix}mask_{index}.png"))

def visualize_image(image, save_dir, index, prefix=""):
    # create directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save images
    save_image(image, os.path.join(save_dir, f"{prefix}image_{index}.png"))

def visualize_image_with_mask(image, mask, save_dir, index, prefix=""):
    # create directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if mask is given in 3 channels format convert it to 1 channel
    if mask.shape[-1] == 3:
        mask = mask[:, :, 0]
   
    image_with_mask = create_image_with_mask(image, mask)
    # save images
    save_image(image_with_mask, os.path.join(save_dir, f"{prefix}image_with_mask_{index}.png"))

def create_image_with_mask(image, mask):
    # check image and mask shapes and types
    assert type(image) == np.ndarray and type(mask) == np.ndarray
    assert image.shape[-1] == 3 and len(mask.shape) == 2 and mask.shape[0] == image.shape[0] and mask.shape[1] == image.shape[1] 
    # get all label ids from gt_mask
    seg = np.zeros(image.shape, dtype=np.uint8)
    for label in labels:
        seg[mask == label.id] = label.color
    return (image * 0.5 + seg * 0.5).astype(np.uint8)

def save_image(image, save_path):
    image = Image.fromarray(image)
    image.save(save_path)

        