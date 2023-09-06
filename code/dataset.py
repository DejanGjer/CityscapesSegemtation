from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter

from labels import labels

class Dataset:
    def __init__(self, checkpoint, to_sample=False, sample_size=80):
        self.chekpoint = checkpoint
        self.to_sample = to_sample
        self.sample_size = sample_size

        self.train_ds, self.validation_ds, self.test_ds = self.load_or_download_dataset()
        if to_sample:
            self.train_ds = self.train_ds.select(range(sample_size))
            self.validation_ds = self.validation_ds.select(range(sample_size))
            self.test_ds = self.test_ds.select(range(sample_size))
        # create lebel2id and id2label dictionaries
        self.label2id = {label.name: label.id for label in labels}
        self.id2label = {label.id: label.name for label in labels}
        print(self.label2id)
        print(self.id2label)
        # load image processor
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        # transform the dataset
        self.train_ds_prepared = self.train_ds.with_transform(self.train_transforms)
        self.validation_ds_prepared = self.validation_ds.with_transform(self.test_transforms)
        self.test_ds_prepared = self.test_ds.with_transform(self.test_transforms)
        example = self.train_ds_prepared[0]
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
    
    def get_dataset(self):
        return self.train_ds, self.validation_ds, self.test_ds
    
    def get_prepared_dataset(self):
        return self.train_ds_prepared, self.validation_ds_prepared, self.test_ds_prepared
    
    def get_num_labels(self):
        return len(self.label2id) - 1 # we subtract one because we ignore -1 label
