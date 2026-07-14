import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import get_dataloaders
from model import AgeRegressionModel
from tqdm import tqdm
import math

def train_model(data_dir, num_epochs=20, batch_size=16, lr=1e-4, save_dir="weights"):
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    model = AgeRegressionModel(pretrained=True).to(device)
    
    # Using L1 Loss (MAE) as it directly translates to years of error
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_mae = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Train
        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, ages = images.to(device), ages.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                loss = criterion(outputs, ages)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train MAE: {train_loss:.4f} years - Val MAE: {val_loss:.4f} years")
        
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_age_model.pth"))
            print(f"--> Saved new best model with Val MAE: {best_val_mae:.4f} years")
            
    # Save last model
    torch.save(model.state_dict(), os.path.join(save_dir, "last_age_model.pth"))
    print(f"Training Complete. Best Validation MAE: {best_val_mae:.4f} years")

if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "data", "dataset ++")
    train_model(data_dir=DATA_DIR, num_epochs=20, batch_size=64, lr=1e-4)
