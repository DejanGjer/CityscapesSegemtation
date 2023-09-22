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
import os
from tqdm import tqdm

import config
from dataset import Dataset
from visualization import visualize_mask, visualize_image, visualize_image_with_mask
from metrics import compute_metrics, add_batch_to_metrics

def visualize_original_image_and_mask(original_image, original_mask, save_dir, index):
    visualize_image(original_image, save_dir, index, "original_")
    visualize_mask(original_mask, save_dir, index, "original_")
    visualize_image_with_mask(original_image, original_mask, save_dir, index, "original_")

def aggregate_logits(logits, labels, original_image, original_mask, dataset, original_image_size, window_width, stride, save_dir, index):
    # this method works only for 1024 x 1024 input images
    aggregated_logits = torch.zeros((logits.shape[1], original_image_size[0], original_image_size[1]))
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).detach().cpu()
    original_image = original_image.detach().cpu().numpy()
    original_mask = dataset.map_training_labels(original_mask.detach().cpu().numpy())[:, :, 0]
    original_mask_to_show = dataset.unmap_training_labels(original_mask)
    visualize_original_image_and_mask(original_image, original_mask_to_show, save_dir, index)
   
    for i, logit_image in enumerate(upsampled_logits):
       aggregated_logits[:, :, i*stride:i*stride+window_width] += logit_image
    aggregated_logits = aggregated_logits.argmax(dim=0).detach().cpu().numpy()
    aggregated_logits_to_show = dataset.unmap_training_labels(aggregated_logits)
    visualize_mask(aggregated_logits_to_show, save_dir, 0, "aggregated_")
    visualize_image_with_mask(original_image, aggregated_logits_to_show, save_dir, 0, "aggregated_")
  
    return original_mask, aggregated_logits

def rescale_logits(logits, original_image, original_mask, dataset, original_image_size, save_dir, index):
    original_image = original_image.detach().cpu().numpy()
    original_mask = dataset.map_training_labels(original_mask.detach().cpu().numpy())[:, :, 0]
    original_mask_to_show = dataset.unmap_training_labels(original_mask)
    visualize_original_image_and_mask(original_image, original_mask_to_show, save_dir, index)
    rescaled_logits = nn.functional.interpolate(
        logits,
        size=original_image_size,
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1).detach().cpu().squeeze().numpy()
    rescaled_logits_to_show = dataset.unmap_training_labels(rescaled_logits)
    visualize_mask(rescaled_logits_to_show, save_dir, 0, "rescaled_")
    visualize_image_with_mask(original_image, rescaled_logits_to_show, save_dir, 0, "rescaled_")

    return original_mask, rescaled_logits
  

def inference(model_checkpoint, dataset, id2label, label2id, device, root_dir):
    inference_ds = dataset.get_inference_dataloader()
    original_validation_ds = dataset.get_original_validation_dataloder()
    model = AutoModelForSemanticSegmentation.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)
    model.to(device)
    model.eval()
    metric = evaluate.load("mean_iou")

    progress_bar = tqdm(range(len(inference_ds)), desc=f"Inference")
    losses = []
    for i, (batch, original_batch) in enumerate(zip(inference_ds, original_validation_ds)):
        input_image = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        original_image = original_batch["image"].squeeze().to(device)
        original_mask = original_batch["semantic_segmentation"].squeeze().to(device)
        with torch.no_grad():
            outputs = model(pixel_values=input_image, labels=labels)
            losses.append(outputs.loss.detach().cpu())
            gt_mask, predicted_mask = None, None
            if dataset.rescale:
                gt_mask, predicted_mask = rescale_logits(outputs.logits, original_image, original_mask, dataset, (1024, 2048),
                                                               os.path.join(root_dir, f"test_{i}"), i)
            else:
                gt_mask, predicted_mask = aggregate_logits(outputs.logits, labels, original_image, original_mask, dataset, (1024, 2048), 
                                config.image_size["width"], config.inference_stride, os.path.join(root_dir, f"test_{i}"), i)
            # print(f"Predection mask: {predicted_mask.shape}")
            # print(f"Original mask: {original_mask.shape}")
            metric.add_batch(
                predictions=np.expand_dims(predicted_mask, axis=0),    
                references=np.expand_dims(gt_mask, axis=0)
            )
        progress_bar.update(1)
    metric_results = compute_metrics(metric, len(id2label))
    metric_results["loss"] = np.array(losses).mean()
    # save results to text file
    with open(os.path.join(root_dir, "results.txt"), "w") as f:
        f.write(str(metric_results))
    print(metric_results)

if __name__ == "__main__":
    dataset = Dataset(config.model_type, config.image_size, config.batch_size, config.rescale, config.to_sample, config.sample_size, config.inference_stride)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference(config.model_checkpoint, dataset, dataset.id2label, dataset.label2id, device, config.save_root_dir)


