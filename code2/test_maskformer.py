from datasets import load_dataset
from transformers import AutoImageProcessor, MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from torchvision.transforms import ColorJitter
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import wandb


from labels import labels

checkpoint = "facebook/maskformer-swin-base-ade"
batch_size = 2
to_sample = True
sample_size = 20

class Dataset:
    def __init__(self, checkpoint, batch_size, to_sample=False, sample_size=80):
        self.chekpoint = checkpoint
        self.batch_size = batch_size
        self.to_sample = to_sample
        self.sample_size = sample_size

        self.train_ds, self.validation_ds, self.test_ds = self.load_or_download_dataset()
        if to_sample:
            self.train_ds = self.train_ds.select(range(sample_size))
            self.validation_ds = self.validation_ds.select(range(sample_size))
            self.test_ds = self.test_ds.select(range(sample_size))
            print("Dataset is sampled")
        # create lebel2id and id2label dictionaries
        self.label2id = {label.name: label.id for label in labels}
        self.id2label = {label.id: label.name for label in labels}
        # load image processor
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        # self.image_processor = MaskFormerFeatureExtractor.from_pretrained(checkpoint)
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        # set format of datasets to torch
        self.train_ds.set_format("torch")
        self.validation_ds.set_format("torch")
        self.test_ds.set_format("torch")
        print(self.train_ds.format)
        # transform the dataset
        self.train_ds_transformed = self.train_ds.with_transform(self.train_transforms)
        self.validation_ds_transformed = self.validation_ds.with_transform(self.test_transforms)
        self.test_ds_transformed = self.test_ds.with_transform(self.test_transforms) 
         
        self.train_dataloader = DataLoader(self.train_ds_transformed, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.validation_dataloader = DataLoader(self.validation_ds_transformed, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(self.test_ds_transformed, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        self.original_trained_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.original_validation_dataloader = DataLoader(self.validation_ds, batch_size=self.batch_size, shuffle=False)
        self.original_test_dataloader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

        example = next(iter(self.train_dataloader))
        ex_pixel_values = example["pixel_values"]
        ex_pixel_mask = example["pixel_mask"]
        ex_mask_labels = example["mask_labels"]
        ex_class_labels = example["class_labels"]
        print("ONE EXAMPLE")
        print(type(ex_pixel_values))
        print(ex_pixel_values.shape)
        print(type(ex_pixel_mask))
        print(ex_pixel_mask.shape)
        for ex in ex_mask_labels:
            print(ex.shape)
        print(type(ex_class_labels))
        for ex in ex_class_labels:
            print(ex.shape)


    def collate_fn(self, batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
        class_labels = [example["class_labels"] for example in batch]
        mask_labels = [example["mask_labels"] for example in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}


    def extract_single_channel(self, image):
        # Split the image into channels (R, G, B)
        r, _, _ = image.split()

        # Create a new single-channel image using the channel you want (e.g., red channel)
        return r

    def train_transforms(self, example_batch):
        images = [self.jitter(x) for x in example_batch["image"]]
        labels = [self.extract_single_channel(x) for x in example_batch["semantic_segmentation"]]
        print("Train transforms")
        print(f"Batch size {len(images)}")
        print(type(images[0]))
        print(type(labels[0]))
        # print all unique labels for each image in the batch
        for i in range(len(images)):
            print(f"Image {i}")
            print(f"Unique labels: {np.unique(labels[i])}")
            print(len(np.unique(labels[i])))

        # print(f"Images shape {images[0].shape}")
        # print(f"Labels shape {labels[0].shape}")
        inputs = self.image_processor(images, labels, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_mask = inputs["pixel_mask"]
        mask_labels = inputs["mask_labels"]
        class_labels = inputs["class_labels"]
        print(type(inputs))
        print(inputs.keys())
        print(type(pixel_values))
        print(f"Pixel values {pixel_values.shape}")
        print(type(pixel_mask))
        print(f"Pixel mask: {pixel_mask.shape}")
        print(type(mask_labels))
        print(type(class_labels))
        print("Mask labels")
        for ex in mask_labels:
            print(ex.shape)
        print("Class labels")
        for ex in class_labels:
            print(ex.shape)
        
        return inputs


    def test_transforms(self, example_batch):
        images = [x for x in example_batch["image"]]
        labels = [self.extract_single_channel(x) for x in example_batch["semantic_segmentation"]]
        inputs = self.image_processor(images, labels)
        return inputs


    def load_or_download_dataset(self):
        dataset = load_dataset("Chris1/cityscapes")
        train_ds = dataset["train"]
        validation_ds = dataset["validation"]
        test_ds = dataset["test"]
        print(train_ds)
        print(validation_ds)
        print(test_ds)
        return train_ds, validation_ds, test_ds
    
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_validation_dataloader(self):
        return self.validation_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader
    
    def get_original_train_dataloader(self):
        return self.original_trained_dataloader
    
    def get_original_validation_dataloder(self):
        return self.original_validation_dataloader
    
    def get_original_test_dataloader(self):
        return self.original_test_dataloader
    
    def get_num_labels(self):
        return len(self.label2id) - 1 # we subtract one because we ignore -1 label
    

if __name__ == "__main__":
    # login to wandb
    api_key = os.environ['WANDB_API_KEY']
    wandb.login(key=api_key)
    run = wandb.init(project="cityscapes_segmentation")
    dataset = Dataset(checkpoint, batch_size, to_sample, sample_size)
    train_ds = dataset.get_train_dataloader()
    validation_ds = dataset.get_validation_dataloader()
    test_ds = dataset.get_test_dataloader()
    num_labels = dataset.get_num_labels()
    print(num_labels)
    print("MODEL")
    model = MaskFormerForInstanceSegmentation.from_pretrained(checkpoint, id2label=dataset.id2label, label2id=dataset.label2id, ignore_mismatched_sizes=True)
    # prepare training device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Model is trained on device {device}")
    # get one example from train set
    example = next(iter(train_ds))
    pixel_values = example["pixel_values"].to(device)
    pixel_mask = example["pixel_mask"].to(device)
    mask_labels = example["mask_labels"]
    for i in range(len(mask_labels)):
        mask_labels[i] = mask_labels[i].to(device)
    class_labels = example["class_labels"]
    for i in range(len(class_labels)):
        class_labels[i] = class_labels[i].to(device)
    
    # get output from model
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels)
    print(outputs)
    wandb.finish()
