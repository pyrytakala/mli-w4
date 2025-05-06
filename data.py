from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import random
import numpy as np
from constants import (
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, DEFAULT_BATCH_SIZE, 
    DEFAULT_TRAIN_SPLIT, MAX_SEQUENCE_LENGTH, START_TOKEN, END_TOKEN,
    RANDOM_SEED
)

def seed_worker(worker_id):
    """Set seed for each worker to ensure basic reproducibility.
    
    Note: This function only sets basic random seeds without enforcing deterministic behavior.
    This provides a good balance between reproducibility and performance.
    """
    worker_seed = RANDOM_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # Note: We intentionally don't set cudnn.deterministic or cudnn.benchmark
    # to maintain better performance while still having basic reproducibility

def flickr_collate_fn(batch, tokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
    """
    Collate function for Flickr dataset that handles tokenization and padding.
    
    Args:
        batch: List of tuples (image, caption)
        tokenizer: CLIP tokenizer instance
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Tuple of (images, input_ids, attention_mask)
        - images: Tensor of shape (batch_size, 3, 224, 224)
        - input_ids: Tensor of shape (batch_size, max_length)
        - attention_mask: Tensor of shape (batch_size, max_length)
    """
    # Stack image tensors into a single batch tensor: (batch_size, 3, 224, 224)
    images = torch.stack([item[0] for item in batch])
    
    # Collect captions into a list of strings
    captions = [item[1] for item in batch]
    
    captions = [START_TOKEN + " " + caption + " " + END_TOKEN for caption in captions]
    
    # Tokenize and pad captions
    tokenized = tokenizer(
        captions,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    
    return images, tokenized.input_ids, tokenized.attention_mask

class FlickrDataset(Dataset):
    def __init__(self, split="test", transform=None, tokenizer=None, max_length: int = MAX_SEQUENCE_LENGTH):
        self.dataset = load_dataset("nlphuji/flickr30k")[split]
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        captions = item["caption"]
        if self.transform:
            image = self.transform(image)
        # Return a random caption from the available ones (each sample has 5)
        caption = random.choice(captions)
        return image, caption

def get_flickr_dataloader(batch_size=DEFAULT_BATCH_SIZE, tokenizer=None, max_length: int = MAX_SEQUENCE_LENGTH):
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
    dataset = FlickrDataset(split="test", transform=preprocess, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: flickr_collate_fn(batch, tokenizer, max_length)
    )

def get_train_val_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE, 
    train_split=DEFAULT_TRAIN_SPLIT, 
    tokenizer=None, 
    max_length: int = MAX_SEQUENCE_LENGTH
):
    full_dataset = get_flickr_dataloader(batch_size=1, tokenizer=tokenizer, max_length=max_length).dataset
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: flickr_collate_fn(batch, tokenizer, max_length),
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: flickr_collate_fn(batch, tokenizer, max_length),
        worker_init_fn=seed_worker
    )
    
    return train_loader, val_loader

def main():
    dataloader = get_flickr_dataloader(batch_size=1)
    images, captions = next(iter(dataloader))
    print(f"Image tensor shape: {images.shape}")
    print(f"Caption: {captions[0]}")

if __name__ == "__main__":
    main()
