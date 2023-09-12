from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
from torch.utils.data import DataLoader

from labels import labels

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
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        # set format of datasets to torch
        self.train_ds.set_format("torch")
        self.validation_ds.set_format("torch")
        self.test_ds.set_format("torch")
        # transform the dataset
        self.train_ds_transformed = self.train_ds.with_transform(self.train_transforms)
        self.validation_ds_transformed = self.validation_ds.with_transform(self.test_transforms)
        self.test_ds_transformed = self.test_ds.with_transform(self.test_transforms) 
         
        self.train_dataloader = DataLoader(self.train_ds_transformed, batch_size=self.batch_size, shuffle=True)
        self.validation_dataloader = DataLoader(self.validation_ds_transformed, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_ds_transformed, batch_size=self.batch_size, shuffle=False)

        self.original_trained_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.original_validation_dataloader = DataLoader(self.validation_ds, batch_size=self.batch_size, shuffle=False)
        self.original_test_dataloader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

        example = next(iter(self.train_dataloader))
        ex_image = example["pixel_values"]
        ex_labels = example["labels"]
        print(type(ex_image))
        print(ex_image.shape)
        print(type(ex_labels))
        print(ex_labels.shape)


    def extract_single_channel(self, image):
        # Split the image into channels (R, G, B)
        r, _, _ = image.split()

        # Create a new single-channel image using the channel you want (e.g., red channel)
        return r

    def train_transforms(self, example_batch):
        images = [self.jitter(x) for x in example_batch["image"]]
        labels = [self.extract_single_channel(x) for x in example_batch["semantic_segmentation"]]
        inputs = self.image_processor(images, labels)
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
