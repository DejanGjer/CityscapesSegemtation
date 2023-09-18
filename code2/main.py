from datasets import load_dataset
from labels import labels, Label
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter, ToTensor
import numpy as np 
import evaluate
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSemanticSegmentation, AdamW, get_scheduler, get_polynomial_decay_schedule_with_warmup
import wandb
import os
import random
from tqdm import tqdm

import config
from dataset import Dataset
from visualization import visualize_samples, visualize_mask

def compute_metrics(metric, num_labels):
    with torch.no_grad():
        metrics = metric.compute(
            num_labels=num_labels,
            ignore_index=0,
            reduce_labels=False
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics
    
def add_batch_to_metrics(metric, logits, labels):
    with torch.no_grad():
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        metric.add_batch(
            predictions=pred_labels,
            references=labels
        )
    return pred_labels
    
def validate(model, metric, eval_ds, id2labels, num_images_to_log, device):
    num_labels = len(id2labels)
    model.eval()
    eval_loss = 0
    progress_bar = tqdm(range(len(eval_ds)), desc=f"Evaluating")
    image_log_dict = {}
    for i, batch in enumerate(eval_ds):
        input_image = batch["pixel_values"].to(device)
        gt_labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=input_image, labels=gt_labels)
            loss = outputs.loss
            eval_loss += loss.item()
            pred_labels = add_batch_to_metrics(metric, outputs.logits, gt_labels)
        if i == 0:
            input_image = input_image.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()
            for i in range(min(gt_labels.shape[0], num_images_to_log)):
                image_log_dict[f"eval/image_{i}"] = wandb.Image(np.transpose(input_image[i], (1, 2, 0)), masks={
                    "predictions" : {
                        "mask_data" : pred_labels[i],
                        "class_labels" : id2labels
                    },
                    "ground_truth" : {
                        "mask_data" : gt_labels[i],
                        "class_labels" : id2labels
                    }
                })
        progress_bar.update(1)
    metric_results = compute_metrics(metric, num_labels)
    print(metric_results)
    result_dict = {
        "eval/loss": eval_loss / len(eval_ds),
        "eval/mIoU": metric_results["mean_iou"],
        "eval/mean_acc": metric_results["mean_accuracy"],
        "eval/overall_acc": metric_results["overall_accuracy"],
    }
    result_dict.update(image_log_dict)
    return result_dict


if __name__ == "__main__":
    # set the seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config.seed)
    np.random.seed(config.seed)

    # login to wandb
    api_key = os.environ['WANDB_API_KEY']
    wandb.login(key=api_key)
    run = wandb.init(project=config.project_name)

    dataset = Dataset(config.model_type, config.image_size, config.batch_size, config.to_sample, config.sample_size)
    train_ds = dataset.get_train_dataloader()
    validation_ds = dataset.get_validation_dataloader()
    # test_ds = dataset.get_test_dataloader()
    num_labels = dataset.get_num_labels()

    model = AutoModelForSemanticSegmentation.from_pretrained(config.model_checkpoint, id2label=dataset.id2label, label2id=dataset.label2id)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    if config.optimier_checkpoint is not None:
        optimizer.load_state_dict(torch.load(config.optimier_checkpoint))
    num_training_steps = config.num_epochs * len(train_ds)
    lr_scheduler = None
    if config.scheduler_type == "linear":
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
    elif config.scheduler_type == "polynomial":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
            lr_end=0.0,
            power=1.0
        )
    if config.lr_scheduler_checkpoint is not None:
        lr_scheduler.load_state_dict(torch.load(config.lr_scheduler_checkpoint))
    metric = evaluate.load("mean_iou")
    # prepare training device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Model is trained on device {device}")

    # training loop
    best_miou = 0
    print("Start training")
    for epoch in range(config.num_epochs):
        epoch_loss_history = []
        progress_bar = tqdm(range(len(train_ds)), desc=f"Epoch {epoch + 1}")
        model.train()
        for i, batch in enumerate(train_ds):
            input_image = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=input_image, labels=labels)
            loss = outputs.loss
            if i % config.train_log_steps == 0:
                wandb.log({"train/loss": loss.item()}, step=epoch * len(train_ds) + i)
            epoch_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        eval_results = validate(model, metric, validation_ds, dataset.id2label, config.eval_images_to_log, device)
        # save model if it is the best one
        if eval_results["eval/mIoU"] > best_miou:
            best_miou = eval_results["eval/mIoU"]
            model.save_pretrained(os.path.join(config.save_root_dir, f"{config.model_type.replace('/', '-')}_{epoch}"))
            torch.save(optimizer.state_dict(), os.path.join(config.save_root_dir, f"{config.model_type.replace('/', '-')}_{epoch}/optimizer.pth"))
            torch.save(lr_scheduler.state_dict(), os.path.join(config.save_root_dir, f"{config.model_type.replace('/', '-')}_{epoch}/lr_scheduler.pth"))
        wandb.log(eval_results, step=(epoch + 1) * len(train_ds) - 1)

    wandb.finish()

