from torchvision import transforms


mean = (0.485, 0.456, 0.406)        # Imagenet style preprocessing
std = (0.229, 0.224, 0.225)
img_size = (224, 224)


def build_train_transform():
    
    transform = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.RandomErasing(p=0.3),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    
    return transform
    

def build_val_transform():
    
    transform = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    
    return transform