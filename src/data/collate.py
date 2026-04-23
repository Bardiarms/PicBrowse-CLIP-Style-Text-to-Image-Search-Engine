import torch
from src.models.tokenizer import WordTokenizer
from typing import Dict, Any


class PicBrowseCollator:
    
    def __init__(self,tokenizer: WordTokenizer, 
                 max_length=32
        ):
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    
    def __call__(self, batch: list)-> Dict[str, Any]:
        
        images = [sample["image"] for sample in batch]
        batch_images = torch.stack(images, dim=0)
        
        captions = [sample["caption"] for sample in batch]
        encoded = [self.tokenizer.encode_plus(c, max_length=self.max_length) for c in captions]
        
        input_ids = torch.tensor([ids["input_ids"] for ids in encoded], dtype=torch.long)
        attention_mask = torch.tensor([masks["attention_mask"] for masks in encoded], dtype=torch.long)
        
        
        file_names = [sample["file_name"] for sample in batch]
        
        return {
            "images": batch_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "captions": captions,
            "file_names": file_names
        }
        


class CachedPicBrowseCollator:
    
    def __init__(self, tokenizer: WordTokenizer,
                max_length = 32        
        ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    
    def __call__(self, batch: list)-> Dict[str, Any]:
        
        image_embeddings = [sample["image_embedding"] for sample in batch]
        batch_image_embeddings = torch.stack(image_embeddings, dim=0)
        
        captions = [sample["caption"] for sample in batch]
        encoded = [self.tokenizer.encode_plus(c, max_length=self.max_length) for c in captions]
        
        input_ids = torch.tensor([ids["input_ids"] for ids in encoded], dtype=torch.long)
        attention_mask = torch.tensor([masks["attention_mask"] for masks in encoded], dtype=torch.long)
        
        
        file_names = [sample["file_name"] for sample in batch]
        
        return {
            "image_embeddings": batch_image_embeddings,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "captions": captions,
            "file_names": file_names
        }