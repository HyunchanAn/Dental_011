import os
import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import AgeRegressionModel
import matplotlib.pyplot as plt

def test_model(data_dir, weights_path, batch_size=8, use_hybrid=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, val_loader = get_dataloaders(data_dir, batch_size=batch_size, use_hybrid=use_hybrid)
    
    model = AgeRegressionModel(pretrained=False)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}. Evaluating with random weights.")
        
    model.to(device)
    model.eval()
    
    criterion = nn.L1Loss()
    val_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in val_loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            loss = criterion(outputs, ages)
            val_loss += loss.item() * images.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
            
    mae = val_loss / len(val_loader.dataset)
    print(f"Validation MAE: {mae:.4f} years")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_preds, alpha=0.6)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Age Estimation Results (MAE: {mae:.2f} yrs)')
    plt.grid(True)
    plt.savefig('evaluation_results.png')
    print("Saved plot to evaluation_results.png")

if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "data", "dataset ++")
    
    # Default to checking for hybrid model first
    WEIGHTS_PATH = os.path.join("weights", "best_hybrid_age_model.pth")
    USE_HYBRID = True
    
    if not os.path.exists(WEIGHTS_PATH):
        WEIGHTS_PATH = os.path.join("weights", "best_age_model.pth")
        USE_HYBRID = False
        
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Weights not found at {WEIGHTS_PATH}. Please train the model first.")
    else:
        test_model(DATA_DIR, WEIGHTS_PATH, batch_size=8, use_hybrid=USE_HYBRID)
