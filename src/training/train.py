from torch.utils.data import DataLoader
import torch
from models.clip_model import PicBrowseModel
from models.loss import ContrastiveLoss
from torch.optim import AdamW
from typing import List


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_one_epoch(clip_model: PicBrowseModel,
                    dataloader: DataLoader,
                    optimizer: torch.optim,
    )-> float: 
    clip_model.train()
    
    running_loss = 0.0
    contrastive_loss = ContrastiveLoss() 
    
    for batch in dataloader:
        images = batch["images"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        
        outputs = clip_model(images=images, input_ids=input_ids, attention_mask=attention_mask)
        logits_per_image = outputs["logits_per_image"]
        logits_per_text = outputs["logits_per_text"]
        
        loss = contrastive_loss(logits_per_image=logits_per_image, logits_per_text=logits_per_text)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    average_loss = running_loss / len(dataloader)
    
    return average_loss
     

def save_best_checkpoint(clip_model: PicBrowseModel,
                    optimizer: torch.optim,
                    epoch: int,
                    best_loss: float,
                    loss: float,
                    save_path: str,
                    
    )-> float: 
    if loss < best_loss:
        checkpoint_dict = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": clip_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        torch.save(checkpoint_dict, save_path)
        best_loss = loss

    return best_loss
        

def train(clip_model: PicBrowseModel,
            dataloader: DataLoader, 
            optimizer: torch.optim,
            epochs: int,
            save_path:str
    )-> List[float]:
    best_loss = 9999999
    loss_history = []
    clip_model.to(DEVICE)
    
    
    for epoch in range(epochs):
        
        average_loss = train_one_epoch(clip_model=clip_model,
                                       dataloader=dataloader,
                                       optimizer=optimizer)
        
        loss_history.append(average_loss)
        
        best_loss = save_best_checkpoint(clip_model=clip_model,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         best_loss=best_loss,
                                         loss=average_loss,
                                         save_path=save_path)
        
        print(f"Epoch {epoch+1}/{epochs} - Current Loss: {average_loss}, Best Loss: {best_loss}")
        
    return loss_history