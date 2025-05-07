import wandb
import os

# Initialize a new W&B run
run = wandb.init(project="image-captioning")  # Replace with your project name

# Upload the checkpoint file
artifact = wandb.Artifact('model_checkpoint', type='model')
artifact.add_file('checkpoints/epoch_20.pth')
run.log_artifact(artifact)

# Finish the run
run.finish()