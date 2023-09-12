from datasets import load_dataset
from labels import labels, Label
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter, ToTensor
import numpy as np 
import evaluate
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSemanticSegmentation, AdamW, get_scheduler
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
            ignore_index=255,
            reduce_labels=False,
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
        metric.add_batch(
            predictions=pred_labels,
            references=labels
        )
    
def validate(model, eval_ds, num_labels):
    model.eval()
    eval_loss = 0
    progress_bar = tqdm(range(len(eval_ds)), desc=f"Evaluating")
    for batch in eval_ds:
        input_image = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=input_image, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
            add_batch_to_metrics(metric, outputs.logits, labels)
        progress_bar.update(1)
    metric_results = compute_metrics(metric, num_labels)
    return {
        "eval/loss": eval_loss / len(eval_ds),
        "eval/mIoU": metric_results["mean_iou"],
        "eval/mean_acc": metric_results["mean_accuracy"],
        "eval/overall_acc": metric_results["overall_accuracy"],
    }

# def prepare_training(learning_rate, bacth_size, num_epochs, train_log_steps, eval_log_steps, 
#                      num_of_checkpoints, save_dir, data_seed):
#     return TrainingArguments(
#         output_dir=save_dir,
#         learning_rate=learning_rate,
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=bacth_size,
#         per_device_eval_batch_size=bacth_size,
#         save_total_limit=num_of_checkpoints,
#         evaluation_strategy="steps",
#         save_strategy="steps",
#         save_steps=eval_log_steps,
#         eval_steps=eval_log_steps,
#         logging_steps=train_log_steps,
#         eval_accumulation_steps=5,
#         remove_unused_columns=False,
#         data_seed = data_seed,
#         report_to="wandb"
#     )


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

    dataset = Dataset(config.checkpoint, config.batch_size, config.to_sample, config.sample_size)
    train_ds = dataset.get_train_dataloader()
    validation_ds = dataset.get_validation_dataloader()
    test_ds = dataset.get_test_dataloader()
    num_labels = dataset.get_num_labels()

    model = AutoModelForSemanticSegmentation.from_pretrained(config.checkpoint, id2label=dataset.id2label, label2id=dataset.label2id)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = config.num_epochs * len(train_ds)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    metric = evaluate.load("mean_iou")
    # prepare training device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Model is trained on device {device}")

    # training loop
    loss_history = []
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
        
        loss_history.append(epoch_loss_history)
        eval_results = validate(model, validation_ds, num_labels)
        wandb.log(eval_results, step=(epoch + 1) * len(train_ds) - 1)

    # training_args = prepare_training(config.learning_rate, config.batch_size, config.num_epochs, config.train_log_steps,
    #                                  config.eval_log_steps, config.num_of_checkpoints, config.save_root_dir, config.data_seed)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_ds,
    #     eval_dataset=validation_ds,
    #     compute_metrics=compute_metrics
    # )
    # trainer.train()
    # trainer.predict(test_ds)


    # model.eval()
    # for example in test_ds:
    #     input_image = example["pixel_values"]
    #     output = model(**example)
    #     print(output.loss, output.logits.shape)

    wandb.finish()

