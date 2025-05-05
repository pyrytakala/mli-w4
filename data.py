from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class FlickrDataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.dataset = load_dataset("nlphuji/flickr30k")[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["caption"]
        if self.transform:
            image = self.transform(image)
        return image, caption


def get_flickr_dataloader(batch_size=1):
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    dataset = FlickrDataset(split="test", transform=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    dataloader = get_flickr_dataloader(batch_size=1)
    images, captions = next(iter(dataloader))
    print(f"Image tensor shape: {images.shape}")
    print(f"Caption: {captions[0]}")

if __name__ == "__main__":
    main()
