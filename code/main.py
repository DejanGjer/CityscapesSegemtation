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

import config
from dataset import Dataset
from visualization import visualize_samples

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        print(logits.shape)
        print(labels.shape)
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        print(logits_tensor.shape)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics

# def extract_single_channel(image):
#     # Split the image into channels (R, G, B)
#     r, _, _ = image.split()

#     # Create a new single-channel image using the channel you want (e.g., red channel)
#     return r

# def train_transforms(example_batch):
#     images = [jitter(x) for x in example_batch["image"]]
#     labels = [extract_single_channel(x) for x in example_batch["semantic_segmentation"]]
#     inputs = image_processor(images, labels)
#     return inputs


# def val_transforms(example_batch):
#     images = [x for x in example_batch["image"]]
#     labels = [extract_single_channel(x) for x in example_batch["semantic_segmentation"]]
#     inputs = image_processor(images, labels)
#     return inputs


# def get_dataset():
#     dataset = load_dataset("Chris1/cityscapes")
#     train_ds = dataset["train"]
#     validation_ds = dataset["validation"]
#     test_ds = dataset["test"]
#     print(train_ds)
#     print(validation_ds)
#     print(test_ds)
#     return train_ds, validation_ds, test_ds

def prepare_training(learning_rate, bacth_size, num_epochs, train_log_steps, eval_log_steps, 
                     num_of_checkpoints, save_dir, data_seed):
    return TrainingArguments(
        output_dir=save_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=bacth_size,
        per_device_eval_batch_size=bacth_size,
        save_total_limit=num_of_checkpoints,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=eval_log_steps,
        eval_steps=eval_log_steps,
        logging_steps=train_log_steps,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
        data_seed = data_seed,
        report_to="wandb"
    )


if __name__ == "__main__":
    # login to wandb
    api_key = os.environ['WANDB_API_KEY']
    wandb.login(key=api_key)
    run = wandb.init(project=config.project_name)
    # train_ds, validation_ds, test_ds = get_dataset()
    # train_ds = train_ds.select(range(80))
    # validation_ds = validation_ds.select(range(16))
    # test_ds = test_ds.select(range(16))
    # create lebel2id and id2label dictionaries
    # label2id = {label.name: label.id for label in labels}
    # id2label = {label.id: label.name for label in labels}
    # print(label2id)
    # print(id2label)
    # # load image processor
    # image_processor = AutoImageProcessor.from_pretrained(config.checkpoint)
    # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    # # transform the dataset
    # train_ds.set_transform(train_transforms)
    # validation_ds.set_transform(val_transforms)
    # test_ds.set_transform(val_transforms)
    # example = train_ds[0]
    # ex_image = example["pixel_values"]
    # ex_labels = example["labels"]
    # print(type(ex_image))
    # print(ex_image.shape)
    # print(type(ex_labels))
    # print(ex_labels.shape)

    dataset = Dataset(config.checkpoint, config.to_sample, config.sample_size)
    train_ds, validation_ds, test_ds = dataset.get_prepared_dataset()
    num_labels = dataset.get_num_labels()
    model = AutoModelForSemanticSegmentation.from_pretrained(config.checkpoint, id2label=dataset.id2label, label2id=dataset.label2id)
    training_args = prepare_training(config.learning_rate, config.batch_size, config.num_epochs, config.train_log_steps,
                                     config.eval_log_steps, config.num_of_checkpoints, config.save_root_dir, config.data_seed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=compute_metrics
    )
    trainer.train()
    preds, gts, test_metrics = trainer.predict(test_ds)
    wandb.log(test_metrics)
    print("TESTING")
    print(preds.shape)
    print(gts.shape)
    visualize_samples(gts, preds, os.path.join(config.save_root_dir, "plots"), True, config.num_inference_samples)
    # logits_tensor = torch.from_numpy(preds)
    # logits_tensor = nn.functional.interpolate(
    #     logits_tensor,
    #     size=gts.shape[-2:],
    #     mode="bilinear",
    #     align_corners=False,
    # ).argmax(dim=1)
    # print(logits_tensor.shape)
    # pred_labels = logits_tensor.detach().cpu().numpy()
    # # randomly chose 10 images from the test set and log them to wandb
    # for i in range(config.num_inference_samples):
    #     index = np.random.randint(0, len(test_ds))
    #     original_image = test_ds[index]["pixel_values"]
    #     # convert original_iamge from (3, 512, 512) to (512, 512, 3)
    #     original_image = np.transpose(original_image, (1, 2, 0))
    #     prediction_mask = pred_labels[index]
    #     ground_truth_mask = test_ds[index]["labels"]
    #     class_labels = id2label
    #     wandb.log(
    #         {f"image_{index}" : wandb.Image(original_image, masks={
    #             "predictions" : {
    #                 "mask_data" : prediction_mask,
    #                 "class_labels" : class_labels
    #             },
    #             "ground_truth" : {
    #                 "mask_data" : ground_truth_mask,
    #                 "class_labels" : class_labels
    #             }
    #         })})
        

    # index = 0
    # original_image = validation_ds[index]["pixel_values"]
    # # convert original_iamge from (3, 512, 512) to (512, 512, 3)
    # original_image = np.transpose(original_image, (1, 2, 0))
    # prediction_mask = pred_labels[index]
    # ground_truth_mask = validation_ds[index]["labels"]
    # class_labels = id2label
    # wandb.log(
    #     {"my_image_key" : wandb.Image(original_image, masks={
    #         "predictions" : {
    #             "mask_data" : prediction_mask,
    #             "class_labels" : class_labels
    #         },
    #         "ground_truth" : {
    #             "mask_data" : ground_truth_mask,
    #             "class_labels" : class_labels
    #         }
    #     })})

    wandb.finish()

