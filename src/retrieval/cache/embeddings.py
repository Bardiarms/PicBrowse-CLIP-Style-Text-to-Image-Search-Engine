from pathlib import Path
import pandas as pd
from src.models.image_encoder import FrozenImageEncoder
from src.data.transforms import build_val_transform
import torch.nn.functional as F
from PIL import Image
import torch
import os

@torch.no_grad()
def build_image_embedding_cache(
    captions_path: str,
    image_folder: str,
    save_path: str,
    batch_size: int = None,
):
    captions_file = Path(captions_path)
    image_folder = Path(image_folder)
    save_path = Path(save_path)
    
    table = pd.read_csv(captions_file)
    unique_image_names = table["image"].astype(str).drop_duplicates().tolist()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_encoder = FrozenImageEncoder().to(device)
    image_encoder.eval()
    
    transform = build_val_transform()
    
    cached_embeddings = {}
    for image_name in unique_image_names:
        
        image_path = image_folder / image_name
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_transform = transform(image).unsqueeze(0).to(device)
        
        embedding = image_encoder(image_transform)
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.squeeze(0).cpu()
        
        cached_embeddings[image_name] = embedding
        
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached_embeddings, save_path)