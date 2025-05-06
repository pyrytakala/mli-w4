import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from model import ImageCaptionModel, set_seed
from data import get_train_val_dataloaders
from torch.utils.data import DataLoader
from constants import (
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_NUM_EPOCHS,
    MAX_SEQUENCE_LENGTH, VALIDATION_FREQUENCY, IMAGE_MEAN, IMAGE_STD,
    RANDOM_SEED
)
import base64
from io import BytesIO
from torchvision.transforms.functional import to_pil_image
import torchvision

def log_val_examples(
    model: ImageCaptionModel,
    val_loader: DataLoader,
    device: torch.device,
    step: int,
    num_examples: int = 4
) -> None:
    """Log validation examples to wandb.
    
    Args:
        model: The model to use for generation
        val_loader: Validation dataloader
        device: Device to run inference on
        step: Current training step
        num_examples: Number of examples to log
    """
    model.eval()
    
    # Get a batch of validation examples
    images, input_ids, attention_mask = next(iter(val_loader))
    images = images.to(device)
    
    # Generate captions
    with torch.no_grad():
        generated_ids = model.generate(images, max_length=MAX_SEQUENCE_LENGTH, temperature=1.0)
    
    # Decode generated captions
    generated_captions = [
        model.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in generated_ids
    ]
    
    # Decode ground truth captions
    ground_truth_captions = [
        model.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in input_ids
    ]
    
    # Create HTML table for side-by-side display
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGE_MEAN, IMAGE_STD)],
        std=[1/s for s in IMAGE_STD]
    )
    html_content = "<table style='width:100%'>"
    for i, (img, gt, pred) in enumerate(zip(images, ground_truth_captions, generated_captions)):
        if i >= num_examples:
            break
        # Denormalize image
        img_disp = inv_normalize(img.cpu())
        img_disp = torch.clamp(img_disp, 0, 1)
        pil_img = to_pil_image(img_disp)
        # Save to buffer
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        # Encode as base64
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        # Create data URI
        img_src = f"data:image/png;base64,{img_b64}"
        # Create table row with image and captions side by side
        html_content += f"""
        <tr>
            <td style='width:300px'>
                <img src='{img_src}' style='max-width:100%'/>
            </td>
            <td style='width:800px; padding-left:10px; vertical-align:top'>
                <p>{pred}</p>
                <p><i>{gt}</i></p>
            </td>
        </tr>
        """
    html_content += "</table>"
    
    # Log to wandb
    wandb.log({
        "val_examples": wandb.Html(html_content),
        "step": step
    })

def validate_epoch(
    model: ImageCaptionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 20  # Limit validation to 20 batches
) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (images, input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Validating")):
            if batch_idx >= max_batches:
                break
                
            # Move data to device
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Create target tokens (shifted by one)
            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            
            # Forward pass
            logits = model(images, input_ids, tgt_mask=None)
            
            # Slice logits to only use caption part
            num_image_patches = logits.shape[1] - input_ids.shape[1]
            caption_logits = logits[:, num_image_patches:, :]
            
            # Calculate loss
            batch_size, seq_len, vocab_size = caption_logits.shape
            
            # Reshape tensors
            logits_reshaped = caption_logits.reshape(-1, vocab_size)
            target_reshaped = target_ids.reshape(-1)
            mask_reshaped = attention_mask.reshape(-1)
            
            # Apply mask to loss calculation
            loss = criterion(logits_reshaped, target_reshaped)
            masked_loss = loss * mask_reshaped
            
            # Update metrics
            total_loss += masked_loss.sum().item()
            total_tokens += mask_reshaped.sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else 0.0

def train_epoch(
    model: ImageCaptionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader = None,
    val_every: int = VALIDATION_FREQUENCY  # Use the constant from constants.py
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (images, input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Training")):
        # Move data to device
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Create target tokens (shifted by one)
        target_ids = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images, input_ids, tgt_mask=None)
        
        # Slice logits to only use caption part
        num_image_patches = logits.shape[1] - input_ids.shape[1]
        caption_logits = logits[:, num_image_patches:, :]
        
        # Calculate loss
        batch_size, seq_len, vocab_size = caption_logits.shape
        
        # Reshape tensors
        logits_reshaped = caption_logits.reshape(-1, vocab_size)
        target_reshaped = target_ids.reshape(-1)
        mask_reshaped = attention_mask.reshape(-1)
        
        # Apply mask to loss calculation
        loss = criterion(logits_reshaped, target_reshaped)
        masked_loss = loss * mask_reshaped
        
        # Backward pass
        masked_loss.mean().backward()
        optimizer.step()
        
        # Update metrics
        total_loss += masked_loss.sum().item()
        total_tokens += mask_reshaped.sum().item()
        
        # Print loss for this batch
        batch_loss = masked_loss.mean().item()
        print(f"Batch loss: {batch_loss:.4f}")
        
        # Log batch metrics to wandb
        wandb.log({
            "batch_loss": batch_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Compute examples and validation loss every val_every batches
        if val_loader is not None and (batch_idx) % val_every == 0:

            log_val_examples(model, val_loader, device, step=batch_idx + 1)

            val_loss = validate_epoch(model, val_loader, criterion, device)
            print(f"Validation Loss (batch {batch_idx}): {val_loss:.4f}")
            wandb.log({
                "val_loss": val_loss,
                "val_step": batch_idx
            })
            
            
    
    return total_loss / total_tokens

def main():
    # Set random seeds for reproducibility
    set_seed(RANDOM_SEED)
    
    # Initialize wandb
    wandb.init(
        project="image-captioning",
        config={
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "num_epochs": DEFAULT_NUM_EPOCHS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = ImageCaptionModel().to(device)
    
    # Log model architecture
    wandb.watch(model)
    
    # Get train and validation dataloaders
    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=DEFAULT_BATCH_SIZE,
        tokenizer=model.tokenizer,
        max_length=MAX_SEQUENCE_LENGTH
    )
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # Training loop
    for epoch in range(DEFAULT_NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{DEFAULT_NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            val_loader=val_loader, val_every=VALIDATION_FREQUENCY
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss
        })
        
        # Log validation examples at the end of each epoch
        log_val_examples(model, val_loader, device, step=(epoch + 1) * len(train_loader))
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 