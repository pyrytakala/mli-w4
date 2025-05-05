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

# Text embedding demo
text_encoder = clip_model.text_model
text_embeddings = text_encoder.embeddings

# Get the first word from the caption
first_word = captions[0][0].split()[0]
print(f"\nFirst word: {first_word}")

# Get the token ID for the first word
tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
token_id = tokenizer.encode(first_word, add_special_tokens=False)[0]

# Get the embedding for the first word
word_embedding = text_embeddings.token_embedding(torch.tensor([token_id]))
print(f"Word embedding shape: {word_embedding.shape}")
print(f"Word embedding: {word_embedding}")
