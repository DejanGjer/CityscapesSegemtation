project_name="cityscapes_segmentation"
checkpoint = "nvidia/mit-b0"

# dataset
to_sample = True
sample_size = 20
image_size = {"height": 512, "width": 512}

learning_rate = 7e-4
batch_size = 2
num_epochs = 2

# logging
train_log_steps = 2
eval_log_steps = 2
eval_images_to_log = 5
seed = 42

# save directories
num_of_checkpoints = 2
save_root_dir = "segformer-b0-cityscapes"

# testing
num_inference_samples = 10
