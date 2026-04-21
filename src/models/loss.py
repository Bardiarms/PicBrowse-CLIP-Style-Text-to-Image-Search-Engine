import torch
import torch.nn.functional as F

class ContrastiveLoss:

    def __call__(self, logits_per_image: torch.Tensor,
                        logits_per_text: torch.Tensor)-> torch.Tensor:
        """
        Compute the contrastive loss (image-to-text + text-to-image).

        Args:
            logits_per_image: similarity scores between images and all texts
            logits_per_text: similarity scores between texts and all images
        
        Returns:
            total_loss: A scalar contrastive loss value
        """
        # Compute image-to-text loss
        labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        
        # Compute text_to_image loss
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)
        
        # Average the two losses
        total_loss = (image_to_text_loss + text_to_image_loss) / 2
        
        return total_loss