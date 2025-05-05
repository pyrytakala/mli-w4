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
import transformers
from typing import Optional, Tuple
from constants import (
    IMAGE_CHANNELS, IMAGE_SIZE, CLIP_HIDDEN_SIZE, EMBEDDING_SIZE,
    NUM_HEADS, FEEDFORWARD_DIM, NUM_DECODER_LAYERS, CLIP_MODEL_NAME, START_TOKEN
)

class ImageCaptionModel(nn.Module):
    def __init__(self, clip_model_name: str = CLIP_MODEL_NAME):
        super().__init__()
        
        # Load CLIP model
        self.clip_model = transformers.CLIPModel.from_pretrained(clip_model_name)
        
        # Image encoder (using all but last layer from CLIP)
        self.image_encoder = self.clip_model.vision_model
        self.image_projection = nn.Linear(CLIP_HIDDEN_SIZE, EMBEDDING_SIZE)
        
        # Text encoder (using CLIP's text model)
        self.text_encoder = self.clip_model.text_model
        self.text_embeddings = self.text_encoder.embeddings
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
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
        
        # Tokenizer
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(clip_model_name)
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP's vision model.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            Image features of shape (batch_size, seq_len, embedding_size)
        """
        # Debug input shape
        print(f"Input images shape: {images.shape}")
        
        with torch.no_grad():
            # CLIP vision model outputs (batch_size, seq_len, hidden_size)
            vision_output = self.image_encoder(images)
            print(f"Vision model output shape: {vision_output.last_hidden_state.shape}")
            image_features = vision_output.last_hidden_state
        
        # Project to (batch_size, seq_len, embedding_size)
        projected_features = self.image_projection(image_features)
        print(f"Projected features shape: {projected_features.shape}")
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
        # Debug input shapes
        print(f"Forward input images shape: {images.shape}")
        if text_tokens is not None:
            print(f"Forward input text_tokens shape: {text_tokens.shape}")
        
        # Encode image: (batch_size, seq_len, embedding_size)
        image_features = self.encode_image(images)
        
        if text_tokens is None:
            # Start with just image features: (batch_size, seq_len, embedding_size)
            decoder_input = image_features
        else:
            # Encode text tokens: (batch_size, seq_len, embedding_size)
            text_features = self.encode_text(text_tokens)
            
            # Ensure batch sizes match
            if text_features.size(0) != image_features.size(0):
                # Repeat text features to match image batch size
                repeat_factor = image_features.size(0) // text_features.size(0)
                if repeat_factor > 0:
                    text_features = text_features.repeat(repeat_factor, 1, 1)
                else:
                    raise ValueError(f"Invalid batch sizes: text_features {text_features.size(0)}, image_features {image_features.size(0)}")
            
            decoder_input = text_features
        
        # Create causal mask if not provided: (seq_len, seq_len)
        if tgt_mask is None:
            seq_len = decoder_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(decoder_input.device)
        
        # Debug shapes
        print(f"decoder_input shape: {decoder_input.shape}")
        print(f"image_features shape: {image_features.shape}")
        print(f"tgt_mask shape: {tgt_mask.shape}")
        
        # Ensure no empty tensors
        if decoder_input.numel() == 0 or image_features.numel() == 0:
            raise ValueError("Empty tensors detected in decoder input or image features")
        
        # Process each batch element separately
        batch_size = decoder_input.size(0)
        decoder_outputs = []
        
        for i in range(batch_size):
            # Get single image and text features
            single_image_features = image_features[i:i+1]  # Keep batch dimension
            single_decoder_input = decoder_input[i:i+1]    # Keep batch dimension
            
            # Decode: (1, seq_len, embedding_size)
            single_output = self.decoder(
                tgt=single_decoder_input,      # (1, seq_len, embedding_size)
                memory=single_image_features,  # (1, seq_len, embedding_size)
                tgt_mask=tgt_mask             # (seq_len, seq_len)
            )
            decoder_outputs.append(single_output)
        
        # Combine outputs: (batch_size, seq_len, embedding_size)
        decoder_output = torch.cat(decoder_outputs, dim=0)
        
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
        
        # Start with image features: (batch_size, seq_len, embedding_size)
        image_features = self.encode_image(images)
        
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
            if (next_token == self.tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]).any():
                break
        
        return tokens


