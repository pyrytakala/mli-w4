# idea:
# generate a caption for an image, one word at a time
# do this by embedding first all the image patches, then all the text tokens

# image embeddings:
#   use all but last layer from CLIP (see clip_emb_demo.py)
#   add one linear projection layer 768 -> 512 (the text embeddings are 512 in dimension)

# text embeddings:
#   use embeddings from CLIP's text model (see clip_emb_demo.py)

# decoder logic:
#     if image patch, encode with image encoder (see clip_emb_demo.py)
#     else, encode with text encoder
#     pass through decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Optional, Tuple
from constants import (
    IMAGE_CHANNELS, IMAGE_SIZE, CLIP_HIDDEN_SIZE, EMBEDDING_SIZE,
    NUM_HEADS, FEEDFORWARD_DIM, NUM_DECODER_LAYERS, CLIP_MODEL_NAME, START_TOKEN, END_TOKEN
)
import os

class DecoderLayer(nn.Module):
    """A single decoder layer with masked self-attention and feedforward network.
    
    This layer implements the standard transformer decoder layer without cross-attention.
    It consists of:
    1. Masked multi-head self-attention
    2. Layer normalization
    3. Position-wise feedforward network
    4. Layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the decoder layer.
        
        Args:
            tgt: Target sequence of shape (batch_size, seq_len, d_model)
            tgt_mask: Optional mask for self-attention of shape (seq_len, seq_len)
            tgt_key_padding_mask: Optional key padding mask of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention block
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt2, tgt2, tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # Feedforward block
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt


class Decoder(nn.Module):
    """Stack of decoder layers."""
    
    def __init__(
        self,
        decoder_layer: DecoderLayer,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all decoder layers.
        
        Args:
            tgt: Target sequence of shape (batch_size, seq_len, d_model)
            tgt_mask: Optional mask for self-attention of shape (seq_len, seq_len)
            tgt_key_padding_mask: Optional key padding mask of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        return output

class ImageCaptionModel(nn.Module):
    def __init__(self, clip_model_name: str = CLIP_MODEL_NAME):
        super().__init__()
        
        self.clip_model = transformers.CLIPModel.from_pretrained(clip_model_name)
        
        # Image encoder (using all but last layer from CLIP)
        self.image_encoder = self.clip_model.vision_model
        self.image_projection = nn.Linear(CLIP_HIDDEN_SIZE, EMBEDDING_SIZE)
        
        # Text encoder (using CLIP's text model)
        self.text_encoder = self.clip_model.text_model
        self.text_embeddings = self.text_encoder.embeddings
        
        self.decoder = Decoder(
            decoder_layer=DecoderLayer(
                d_model=EMBEDDING_SIZE,
                nhead=NUM_HEADS,
                dim_feedforward=FEEDFORWARD_DIM,
                batch_first=True
            ),
            num_layers=NUM_DECODER_LAYERS
        )
        
        # Output layer: project from embedding dim to vocab size
        vocab_size = self.text_embeddings.token_embedding.weight.shape[0]
        self.output_layer = nn.Linear(EMBEDDING_SIZE, vocab_size)
        
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(clip_model_name)
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP's vision model.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)

        Returns:
            Image features of shape (batch_size, seq_len, embedding_size)
        """

        with torch.no_grad():
            # CLIP vision model outputs (batch_size, seq_len, hidden_size)
            vision_output = self.image_encoder(images)
            image_features = vision_output.last_hidden_state

        # Project to (batch_size, seq_len, embedding_size)
        projected_features = self.image_projection(image_features)
        return projected_features

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text tokens using CLIP's text model.
        
        Args:
            text_tokens: Input tokens of shape (batch_size, seq_len)
        
        Returns:
            Text embeddings of shape (batch_size, seq_len, embedding_size)
        """
        return self.text_embeddings(text_tokens)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for generating captions.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
            text_tokens: Optional text tokens for teacher forcing, shape (batch_size, seq_len)
            tgt_mask: Optional mask for the decoder attention, shape (seq_len, seq_len)
            
        Returns:
            Logits for next token prediction, shape (batch_size, seq_len, vocab_size)
        """
        # Encode image: (batch_size, seq_len, embedding_size)
        image_features = self.encode_image(images)
        
        if text_tokens is None:
            # Start with just image features: (batch_size, seq_len, embedding_size)
            decoder_input = image_features
        else:
            # Encode text tokens: (batch_size, seq_len, embedding_size)
            text_features = self.encode_text(text_tokens)
            
            # Ensure batch sizes match
            assert text_features.size(0) == image_features.size(0), f"Text features batch size {text_features.size(0)} != image features batch size {image_features.size(0)}"
            
            # Concatenate image and text features along sequence dimension
            decoder_input = torch.cat([image_features, text_features], dim=1)
        
        # Create causal mask if not provided: (seq_len, seq_len)
        if tgt_mask is None:
            seq_len = decoder_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(decoder_input.device)

        # Ensure no empty tensors
        if decoder_input.numel() == 0:
            raise ValueError("Empty tensors detected in decoder input")
        
        # Process sequence through decoder
        decoder_output = self.decoder(
            tgt=decoder_input,      # (batch_size, seq_len, embedding_size)
            tgt_mask=tgt_mask      # (seq_len, seq_len)
        )
        
        # Output layer: (batch_size, seq_len, vocab_size)
        logits = self.output_layer(decoder_output)
        return logits

    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 20,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate captions for images.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
            max_length: Maximum length of generated captions
            temperature: Temperature for sampling
            
        Returns:
            Generated token IDs of shape (batch_size, seq_len)
        """
        batch_size = images.shape[0]
        device = images.device
        
        
        # Initialize with start token: (batch_size, 1)
        start_token = self.tokenizer.encode(START_TOKEN, add_special_tokens=False)[0]
        tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Create causal mask: (current_seq_len, current_seq_len)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1]).to(device)
            
            # Get next token logits: (batch_size, 1, vocab_size)
            logits = self.forward(images, tokens, tgt_mask)
            next_token_logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            
            # Sample next token: (batch_size, 1)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to tokens: (batch_size, current_seq_len + 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if we hit end token
            if (next_token == self.tokenizer.encode(END_TOKEN, add_special_tokens=False)[0]).any():
                break
        
        return tokens

def save_checkpoint(model, optimizer, epoch, batch_idx, path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if not os.path.exists(path):
        print("No checkpoint found at", path)
        return 0, 0  # start from scratch
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path}")
    return checkpoint["epoch"], checkpoint["batch_idx"]


