import pytest
import torch
from data import get_flickr_dataloader
from constants import IMAGE_CHANNELS, IMAGE_SIZE, DEFAULT_BATCH_SIZE


def test_get_minibatch():
    try:
        # Get a dataloader with default batch size
        dataloader = get_flickr_dataloader(batch_size=DEFAULT_BATCH_SIZE)
        
        # Get one batch
        images, captions = next(iter(dataloader))
        
        # Verify the shapes and types
        assert isinstance(images, torch.Tensor), "Images should be a torch.Tensor"
        assert images.shape == (DEFAULT_BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), f"Expected shape ({DEFAULT_BATCH_SIZE}, {IMAGE_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}), got {images.shape}"
        
        # Verify image values are normalized (allowing for some numerical precision issues)
        assert -2.0 <= images.min() <= 0.0, f"Min value {images.min()} outside expected range"
        assert 0.0 <= images.max() <= 2.5, f"Max value {images.max()} outside expected range"
        
        # Verify captions
        assert isinstance(captions, list), "Captions should be a list"
        assert len(captions) == DEFAULT_BATCH_SIZE, f"Expected {DEFAULT_BATCH_SIZE} captions, got {len(captions)}"
        for i, caption in enumerate(captions):
            assert isinstance(caption, str), f"Caption {i} is not a string"
            assert len(caption) > 0, f"Caption {i} is empty"
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 