from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import random
from constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, DEFAULT_BATCH_SIZE, DEFAULT_TRAIN_SPLIT

def flickr_collate_fn(batch):
    # Stack image tensors into a single batch tensor: (batch_size, 3, 224, 224)
    images = torch.stack([item[0] for item in batch])
    # Collect captions into a list of strings
    captions = [item[1] for item in batch]
    return images, captions

class FlickrDataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.dataset = load_dataset("nlphuji/flickr30k")[split]
        self.transform = transform

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


def get_flickr_dataloader(batch_size=DEFAULT_BATCH_SIZE):
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
    dataset = FlickrDataset(split="test", transform=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=flickr_collate_fn)


def get_train_val_dataloaders(batch_size=DEFAULT_BATCH_SIZE, train_split=DEFAULT_TRAIN_SPLIT):
    """Get train and validation dataloaders for the Flickr dataset.
    
    Args:
        batch_size (int): Batch size for the dataloaders
        train_split (float): Proportion of data to use for training
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get the full dataset
    full_dataset = get_flickr_dataloader(batch_size=1).dataset
    
    # Split into train and validation sets
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders using the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=flickr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=flickr_collate_fn)
    
    return train_loader, val_loader


def main():
    dataloader = get_flickr_dataloader(batch_size=1)
    images, captions = next(iter(dataloader))
    print(f"Image tensor shape: {images.shape}")
    print(f"Caption: {captions[0]}")

if __name__ == "__main__":
    main()
