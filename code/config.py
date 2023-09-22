project_name="cityscapes_segmentation"
model_type = "nvidia/mit-b0"
model_checkpoint = "nvidia/mit-b0"
optimier_checkpoint = None
lr_scheduler_checkpoint = None

# dataset
to_sample = True
sample_size = 10
image_size = {"height": 1024, "width": 1024}
rescale = True # whether to rescale or random crop input images 

# training
learning_rate = 7e-4
batch_size = 2
num_epochs = 1
scheduler_type = "linear"

# logging
train_log_steps = 2
eval_log_steps = 2
eval_images_to_log = 5
seed = 42

# save directories
num_of_checkpoints = 2
save_root_dir = "segformer-b0-cityscapes"

# inference
inference_stride = 512
