import torch
from torch import nn


class TextEncoder(nn.Module):
    
    def __init__(self, vocab_size: int, max_length=32,
                 embed_dim=256, num_heads=4, num_layers=2,
                 ff_dim=512, output_dim=512):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        
        # Token Embedding
        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        
        # Positional Embedding
        self.position_embedding = nn.Embedding(num_embeddings=self.max_length, embedding_dim=self.embed_dim)
        
        # TransformerEncoderLayer
        transformer_layer = nn.TransformerEncoderLayer(
                                d_model=self.embed_dim,
                                nhead=self.num_heads,
                                dim_feedforward=self.ff_dim,
                                batch_first=True
                            )
        
        # Two layers of transformers
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=self.num_layers)
        
        # Output Projection [B, 256] -> [B, 512]
        self.projection = nn.Linear(self.embed_dim, self.output_dim)
        
    
    def forward(self, input_ids, attention_mask):
        
        token_x = self.token_embedding(input_ids)
        
        positions = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pos_x = self.position_embedding(positions)
        
        x = token_x + pos_x
        
        padding_mask = (attention_mask == 0)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        
        pooled_out = self.masked_mean_pool(out, attention_mask)
        
        final_out = self.projection(pooled_out)
        
        return final_out
    
        
    def masked_mean_pool(self, x, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(x)
        masked_x = x * mask
        summed = masked_x.sum(dim=1)
        counts = mask.sum(dim=1)
        counts = counts.clamp(min=1e-9)
        pooled = summed / counts
        
        return pooled
    