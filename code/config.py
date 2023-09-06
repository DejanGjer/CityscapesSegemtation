project_name="cityscapes_segmentation"
checkpoint = "nvidia/mit-b0"

learning_rate = 7e-4
batch_size = 24
num_epochs = 6

train_log_steps = 50
eval_log_steps = 50
data_seed = 42

# save directories
num_of_checkpoints = 2
save_root_dir = "segformer-b0-cityscapes"

# testing
num_inference_samples = 10
