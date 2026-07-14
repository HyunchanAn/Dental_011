import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class DentalAgeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        
        # Extract age from filename: "17 47.jpg" -> age = 17
        self.ages = []
        for fp in file_paths:
            basename = os.path.basename(fp)
            age = float(basename.split(' ')[0])
            self.ages.append(age)
            
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        age = self.ages[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(age, dtype=torch.float32)

def get_dataloaders(data_dir, batch_size=8, img_size=(224, 448)):
    all_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    # Filter files that contain space (valid format)
    valid_files = [f for f in all_files if ' ' in os.path.basename(f)]
    
    # Split 80/20 train/val
    train_files, val_files = train_test_split(valid_files, test_size=0.2, random_state=42)
    
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = DentalAgeDataset(train_files, transform=train_transform)
    val_dataset = DentalAgeDataset(val_files, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader
