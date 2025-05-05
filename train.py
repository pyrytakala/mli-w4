import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from model import ImageCaptionModel
from data import get_train_val_dataloaders
from torch.utils.data import DataLoader
from constants import (
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_NUM_EPOCHS,
    MAX_CAPTION_LENGTH
)

def validate_epoch(
    model: ImageCaptionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            
            # Tokenize captions
            tokenized = model.tokenizer(
                captions,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CAPTION_LENGTH
            )
            input_ids = tokenized.input_ids.to(device)
            
            # Create target tokens (shifted by one)
            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            
            # Create attention mask
            attention_mask = tokenized.attention_mask[:, :-1].to(device)
            
            # Forward pass
            logits = model(images, input_ids, tgt_mask=None)
            
            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            
            # Reshape tensors
            logits_reshaped = logits.reshape(-1, vocab_size)
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
    val_every: int = 100
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc="Training")):
        # Move data to device
        images = images.to(device)
        
        # Tokenize captions (captions is a list of strings)
        tokenized = model.tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CAPTION_LENGTH
        )
        input_ids = tokenized.input_ids.to(device)
        
        # Create target tokens (shifted by one)
        target_ids = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]
        
        # Create attention mask
        attention_mask = tokenized.attention_mask[:, :-1].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images, input_ids, tgt_mask=None)
        
        # Calculate loss
        # Reshape tensors to be contiguous before loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        print(f"Batch size: {batch_size}, Seq len: {seq_len}, Vocab size: {vocab_size}")
        
        # Ensure shapes match before reshaping
        assert target_ids.shape[0] == batch_size, f"Target batch size {target_ids.shape[0]} != logits batch size {batch_size}"
        assert target_ids.shape[1] == seq_len, f"Target seq len {target_ids.shape[1]} != logits seq len {seq_len}"
        assert attention_mask.shape[0] == batch_size, f"Mask batch size {attention_mask.shape[0]} != logits batch size {batch_size}"
        assert attention_mask.shape[1] == seq_len, f"Mask seq len {attention_mask.shape[1]} != logits seq len {seq_len}"
        
        # Reshape tensors
        logits_reshaped = logits.reshape(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        target_reshaped = target_ids.reshape(-1)          # [batch_size * seq_len]
        mask_reshaped = attention_mask.reshape(-1)        # [batch_size * seq_len]
        
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
        
        # Compute validation loss every val_every batches
        if val_loader is not None and (batch_idx + 1) % val_every == 0:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            print(f"Validation Loss (batch {batch_idx + 1}): {val_loss:.4f}")
            wandb.log({
                "val_loss": val_loss,
                "val_step": batch_idx + 1
            })
    
    return total_loss / total_tokens

def main():
    # Initialize wandb
    wandb.init(
        project="image-captioning",
        config={
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "num_epochs": DEFAULT_NUM_EPOCHS,
            "max_caption_length": MAX_CAPTION_LENGTH
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
    train_loader, val_loader = get_train_val_dataloaders(batch_size=DEFAULT_BATCH_SIZE)
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # Training loop
    for epoch in range(DEFAULT_NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{DEFAULT_NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            val_loader=val_loader, val_every=100
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss
        })
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 