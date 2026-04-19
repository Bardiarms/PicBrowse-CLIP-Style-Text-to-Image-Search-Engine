from image_encoder import FrozenImageEncoder
from text_encoder import TextEncoder
from torch import nn
import torch.nn.functional as F
import torch



class PicBrowseModel(nn.Module):
    
    def __init__(self, image_encoder: FrozenImageEncoder,
                 text_encoder: TextEncoder,
                 init_logit_scale = 0.0
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        
    
    # Image embedding computation and normalization    
    def encode_image(self, images):
        image_embeddings = self.image_encoder(images)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
       
        return image_embeddings
    
    
    # Text embedding computation and normalization
    def encode_text(self, input_ids, attention_mask): 
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        return text_embeddings
    
    
    def forward(self, images, input_ids, attention_mask):
        
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Similarity matrix computation
        similarity_matrix = image_embeddings @ text_embeddings.T
        
        # Scale logits
        logits = self.logit_scale.exp() * similarity_matrix

        
        output = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "similarity_matrix": similarity_matrix,
            "logits": logits,
            "logits_per_image": logits,
            "logits_per_text": logits.T
        }
        
        return output