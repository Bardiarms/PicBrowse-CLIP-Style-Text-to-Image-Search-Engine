from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
import os
import errno


class PicBrowseDataset(Dataset):
    
    def __init__(self, csv_file, image_folder, transform):
        self.csv_file = Path(csv_file)
        self.image_folder = Path(image_folder)
        self.transform = transform
        
        self.table = pd.read_csv(self.csv_file)
        
    def __len__(self) -> int:
        return len(self.table)
    
    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        
        file_name = row["image"]
        caption = row["caption"]
        
        caption = str(caption).strip()
        
        
        
        image_path = self.image_folder / file_name
        if not os.path.isfile(image_path):
            raise FileNotFoundError
            
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        sample = {
            "image": image,
            "caption": caption,
            "file_name": file_name
        }
        
        return sample