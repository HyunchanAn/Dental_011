import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders
from model import AgeRegressionModel
from tqdm import tqdm
import math

def train_model(data_dir, num_epochs=20, batch_size=16, lr=1e-4, save_dir="weights", use_hybrid=False):
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size, use_hybrid=use_hybrid)
    
    model = AgeRegressionModel(pretrained=True).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Using ReduceLROnPlateau instead of StepLR
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_mae = float('inf')
    early_stop_patience = 20
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, ages = images.to(device), ages.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                loss = criterion(outputs, ages)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Scheduler steps on val_loss
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train MAE: {train_loss:.4f} - Val MAE: {val_loss:.4f}")
        
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            epochs_no_improve = 0
            model_name = "best_hybrid_age_model.pth" if use_hybrid else "best_age_model.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
            print(f"--> Saved new best model with Val MAE: {best_val_mae:.4f} years")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
    last_name = "last_hybrid_age_model.pth" if use_hybrid else "last_age_model.pth"
    torch.save(model.state_dict(), os.path.join(save_dir, last_name))
    print(f"Training Complete. Best Validation MAE: {best_val_mae:.4f} years")

if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "data", "dataset ++")
    train_model(data_dir=DATA_DIR, num_epochs=150, batch_size=8, lr=1e-4, use_hybrid=True)
