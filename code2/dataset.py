import albumentations as A
from datasets import load_dataset
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
import datetime

from labels import labels
from visualization import visualize_image, visualize_mask, visualize_image_with_mask

class Dataset:
    def __init__(self, checkpoint, image_size, batch_size, to_sample=False, sample_size=80, inference_stride=512):
        self.chekpoint = checkpoint
        self.image_size = image_size
        self.batch_size = batch_size
        self.to_sample = to_sample
        self.sample_size = sample_size
        self.inference_stride = inference_stride

        self.train_ds, self.validation_ds, self.test_ds = self.load_or_download_dataset()
        if to_sample:
            self.train_ds = self.train_ds.select(range(sample_size))
            self.validation_ds = self.validation_ds.select(range(sample_size))
            self.test_ds = self.test_ds.select(range(sample_size))
            print("Dataset is sampled")
        # print one example from test_ds
        # print("Example from dataset class")
        # example = next(iter(self.test_ds))
        # ex_image = np.array(example["image"])
        # ex_labels = np.array(example["semantic_segmentation"])
        # print(ex_image.shape)
        # print(ex_labels.shape)
        # print(np.unique(ex_labels))
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
        # self.image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_resize=True, size=self.image_size, reduce_labels=False)
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_resize=False, reduce_labels=False)
        # set format of datasets to torch
        self.train_ds.set_format("torch")
        self.validation_ds.set_format("torch")
        # self.test_ds.set_format("torch")
        # transform the dataset
        self.train_ds_transformed = self.train_ds.with_transform(self.train_transforms)
        self.validation_ds_transformed = self.validation_ds.with_transform(self.train_transforms)
        # self.test_ds_transformed = self.test_ds.with_transform(self.test_transforms) 
         
        self.train_dataloader = DataLoader(self.train_ds_transformed, batch_size=self.batch_size, shuffle=True)
        self.validation_dataloader = DataLoader(self.validation_ds_transformed, batch_size=self.batch_size, shuffle=False)
        # self.test_dataloader = DataLoader(self.test_ds_transformed, batch_size=self.batch_size, shuffle=False)

        self.original_trained_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.original_validation_dataloader = DataLoader(self.validation_ds, batch_size=self.batch_size, shuffle=False)
        # self.original_test_dataloader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

        # print("Validation example from dataset class")
        # example = next(iter(self.validation_dataloader))
        # ex_image = example["pixel_values"]
        # ex_labels = example["labels"]
        # print(type(ex_image))
        # print(ex_image.shape)
        # print(type(ex_labels))
        # print(ex_labels.shape)
        # print(torch.unique(ex_labels[0]))


    # def extract_single_channel(self, image):
    #     # Split the image into channels (R, G, B)
    #     r, _, _ = image.split()

    #     # Create a new single-channel image using the channel you want (e.g., red channel)
    #     return r

    def extract_single_channel(self, image):
        return image[:, :, 0]

    def train_transforms(self, example_batch):
        # transforms = torchvision.transforms.Compose([
        #     # torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        #     RandomResize(min_scale=0.5, max_scale=2.0),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.RandomCrop((self.image_size["height"], self.image_size["width"]))
        # ])
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
        
        # dir = f"segformer-b0-cityscapes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # for i in range(len(images)):
        #     image = images[i]
        #     label = labels[i]
        #     print(image.shape)
        #     print(label.shape)
        #     visualize_image(np.array(example_batch["image"][i]), dir, i, prefix="original_")
        #     visualize_mask(np.array(example_batch["semantic_segmentation"][i]), dir, i, prefix="original_")
        #     visualize_image_with_mask(np.array(example_batch["image"][i]), np.array(example_batch["semantic_segmentation"][i]), 
        #                               dir, i, prefix="original_")
        #     visualize_image(image, dir, i)
        #     visualize_mask(label, dir, i)
        #     visualize_image_with_mask(image, label, dir, i)
        # print("IMAGES VISUALIZED")
            
        # images = [np.array(transforms(x)) for x in example_batch["image"]]
        labels = [self.map_training_labels(self.extract_single_channel(x)) for x in labels]
        inputs = self.image_processor(images, labels)
        return inputs

    def test_transforms(self, example_batch):
        images = [np.array(x) for x in example_batch["image"]]
        labels = [self.map_training_labels(self.extract_single_channel(x)) for x in example_batch["semantic_segmentation"]]
        inputs = self.image_processor(images, labels)
        return inputs

    def map_label(self, label):
        label_name = self.original_id2label[label]
        if label_name in self.label2id:
            return self.label2id[label_name]
        else:
            return 0

    def map_training_labels(self, labels):
        labels = np.array(labels)
        return np.vectorize(self.map_label)(labels)

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
    
    # def get_test_dataloader(self):
    #     return self.test_dataloader
    
    def get_original_train_dataloader(self):
        return self.original_trained_dataloader
    
    def get_original_validation_dataloder(self):
        return self.original_validation_dataloader
    
    # def get_original_test_dataloader(self):
    #     return self.original_test_dataloader
    
    def get_num_labels(self):
        return len(self.label2id)


class RandomResize(object):
    def __init__(self, min_scale=0.5, max_scale=2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image):
        # Generate a random scale factor within the specified range
        scale_factor = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale
        
        # Apply the random resizing to the image
        new_width = int(image.size[0] * scale_factor)
        new_height = int(image.size[1] * scale_factor)
        resized_image = torchvision.transforms.Resize((new_width, new_height))(image)
        
        return resized_image