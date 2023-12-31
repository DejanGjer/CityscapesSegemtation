import albumentations as A
from datasets import load_dataset
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import ColorJitter
import datetime

from labels import labels


class Dataset:
    def __init__(self, checkpoint, image_size, batch_size, rescale, to_sample=False, sample_size=80, inference_stride=512):
        self.chekpoint = checkpoint
        self.image_size = image_size
        self.batch_size = batch_size
        self.rescale = rescale
        self.to_sample = to_sample
        self.sample_size = sample_size
        self.inference_stride = inference_stride

        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

        self.train_ds, self.validation_ds, self.test_ds = self.load_or_download_dataset()
        if to_sample:
            self.train_ds = self.train_ds.select(range(sample_size))
            self.validation_ds = self.validation_ds.select(range(sample_size))
            self.test_ds = self.test_ds.select(range(sample_size))
            print("Dataset is sampled")

        # create lebel2id and id2label dictionaries
        self.label2id = {label.name: label.id for label in labels}
        self.id2label = {label.id: label.name for label in labels}
        # create lebel2id and id2label dictionaries
        self.original_label2id = {label.name: label.id for label in labels}
        self.original_id2label = {label.id: label.name for label in labels}
        # get labels that we want to use for training and validation
        self.selected_classes = [label for label in labels if label.ignoreInEval == False]
        self.id2label = {0: "other"}
        self.label2id = {"other": 0}
        self.id2label.update({i + 1: label.name for i, label in enumerate(self.selected_classes)})
        self.label2id.update({label.name: i + 1 for i, label in enumerate(self.selected_classes)})
        # load image processor
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_resize=False, reduce_labels=False)
        self.image_processor_with_rescale = AutoImageProcessor.from_pretrained(checkpoint, do_resize=True, size=self.image_size, reduce_labels=False)
        # set format of datasets to torch
        self.train_ds.set_format("torch")
        self.validation_ds.set_format("torch")
        # transform the dataset
        self.train_ds_transformed, self.validation_ds_transformed, self.inference_ds_transformed = None, None, None
        if self.rescale:
            self.train_ds_transformed = self.train_ds.with_transform(self.train_transforms_rescaled)
            self.validation_ds_transformed = self.validation_ds.with_transform(self.test_transforms_rescaled)
            self.inference_ds_transformed = self.validation_ds.with_transform(self.test_transforms_rescaled)
        else:
            self.train_ds_transformed = self.train_ds.with_transform(self.train_transforms)
            self.validation_ds_transformed = self.validation_ds.with_transform(self.train_transforms)
            self.inference_ds_transformed = self.validation_ds.with_transform(self.inference_transforms)
         
        self.train_dataloader = DataLoader(self.train_ds_transformed, batch_size=self.batch_size, shuffle=True)
        self.validation_dataloader = DataLoader(self.validation_ds_transformed, batch_size=self.batch_size, shuffle=False)
        self.inference_dataloader = DataLoader(self.inference_ds_transformed, batch_size=1, shuffle=False)
        # self.test_dataloader = DataLoader(self.test_ds_transformed, batch_size=self.batch_size, shuffle=False)

        self.original_trained_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.original_validation_dataloader = DataLoader(self.validation_ds, batch_size=1, shuffle=False)
        
    def extract_single_channel(self, image):
        return image[:, :, 0]

    def train_transforms(self, example_batch):
        transforms = A.Compose([
            A.HorizontalFlip(),
            A.RandomCrop(width=self.image_size["width"], height=self.image_size["height"])
        ])
        images, labels = [], []
        for i in range(len(example_batch["image"])):
            image = np.array(example_batch["image"][i])
            label = np.array(example_batch["semantic_segmentation"][i])
            transformed = transforms(image=image, mask=label)
            images.append(transformed["image"])
            labels.append(transformed["mask"])
       
        labels = [self.map_training_labels(self.extract_single_channel(x)) for x in labels]
        inputs = self.image_processor(images, labels)
        return inputs
    
    def train_transforms_rescaled(self, example_batch):
        images = [np.array(self.jitter(x)) for x in example_batch["image"]]
        labels = [self.map_training_labels(self.extract_single_channel(np.array(x))) for x in example_batch["semantic_segmentation"]]
        inputs = self.image_processor_with_rescale(images, labels)
        return inputs


    def test_transforms_rescaled(self, example_batch):
        images = [np.array(x) for x in example_batch["image"]]
        labels = [self.map_training_labels(self.extract_single_channel(np.array(x))) for x in example_batch["semantic_segmentation"]]
        inputs = self.image_processor_with_rescale(images, labels)
        return inputs
    
    def divide_image_sliding_window(self, image, window_width, stride):
        _, width, _ = image.shape
        windows = []
        for x in range(0, width - window_width + 1, stride):
            windows.append(image[:, x:x + window_width])
        return windows

    def inference_transforms(self, example_batch):
        # batch size needs to be 1
        assert len(example_batch["image"]) == 1

        images, labels = [], []
        dir = f"segformer-b0-cityscapes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for i in range(len(example_batch["image"])):
            image = np.array(example_batch["image"][i])
            label = np.array(example_batch["semantic_segmentation"][i])
            images.extend(self.divide_image_sliding_window(image, self.image_size["width"], self.inference_stride))
            labels.extend(self.divide_image_sliding_window(label, self.image_size["width"], self.inference_stride))

        labels = [self.map_training_labels(self.extract_single_channel(x)) for x in labels]
        inputs = self.image_processor(images, labels)
        return inputs

    def map_label(self, label):
        label_name = self.original_id2label[label]
        if label_name in self.label2id:
            return self.label2id[label_name]
        else:
            return 0
        
    def unmap_label(self, label):
        label_name = self.id2label[label]
        if label_name in self.original_label2id:
            return self.original_label2id[label_name]
        else:
            return 0
        
    def map_training_labels(self, labels):
        labels = np.array(labels)
        return np.vectorize(self.map_label)(labels)
    
    def unmap_training_labels(self, labels):
        labels = np.array(labels)
        return np.vectorize(self.unmap_label)(labels)

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
    
    def get_inference_dataloader(self):
        return self.inference_dataloader
    
    def get_original_train_dataloader(self):
        return self.original_trained_dataloader
    
    def get_original_validation_dataloder(self):
        return self.original_validation_dataloader
    
    def get_num_labels(self):
        return len(self.label2id)
