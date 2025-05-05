import pytest
import torch
from data import get_flickr_dataloader


def test_get_minibatch():
    try:
        batch_size = 8  # Using a larger batch size
        # Get a dataloader with batch size 8
        dataloader = get_flickr_dataloader(batch_size=batch_size)
        
        # Get one batch
        images, captions = next(iter(dataloader))
        
        # Verify the shapes and types
        assert isinstance(images, torch.Tensor), "Images should be a torch.Tensor"
        assert images.shape == (batch_size, 3, 224, 224), f"Expected shape ({batch_size}, 3, 224, 224), got {images.shape}"
        
        # Verify image values are normalized (allowing for some numerical precision issues)
        assert -2.0 <= images.min() <= 0.0, f"Min value {images.min()} outside expected range"
        assert 0.0 <= images.max() <= 2.5, f"Max value {images.max()} outside expected range"
        
        # Verify captions
        assert isinstance(captions, list), "Captions should be a list"
        assert len(captions) == batch_size, f"Expected {batch_size} captions, got {len(captions)}"
        for i, caption in enumerate(captions):
            assert isinstance(caption, str), f"Caption {i} is not a string"
            assert len(caption) > 0, f"Caption {i} is empty"
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 