import transformers
from data import get_flickr_dataloader
import torch

clip_model = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
image_encoder = clip_model.vision_model

dataloader = get_flickr_dataloader(batch_size=1)
images, captions = next(iter(dataloader))

with torch.no_grad():
    output = image_encoder(images)
    last_hidden_state = output.last_hidden_state

print(f"Last hidden state shape: {last_hidden_state.shape}")
print(f"Last hidden state (first element): {last_hidden_state[0]}")
print(f"Caption: {captions[0]}")
