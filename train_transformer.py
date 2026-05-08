import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.data.data_manager import DataManager
from src.models.price_predictor import PriceTransformer
from src.models.dataset import create_sequences

def train():
    print("[1/4] Fetching Market Data...")
    dm = DataManager()
    
    # Using KuCoin for deep historical access (6000 limit)
    df = dm.get_crypto_data("kucoin", "BTC/USDT", "1h", 6000)
    
    print(f"[INFO] Retrieved {len(df)} historical bars from DataManager.")
    
    if len(df) < 100:
        print("[ERROR] Not enough data fetched to train. Exiting.")
        return

    # Normalize Data
    data_values = df.iloc[:, :10].values 
    data_mean = np.mean(data_values, axis=0)
    data_std = np.std(data_values, axis=0) + 1e-8
    normalized_data = (data_values - data_mean) / data_std

    print("[2/4] Building Sequences (Seq Length: 50)...")
    
    X, y = create_sequences(normalized_data, seq_len=50, threshold=0.005)
    
    # Calculate Class Weights based on entire dataset
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"      Label Distribution: Down(0): {dist.get(0,0)}, Neutral(1): {dist.get(1,0)}, Up(2): {dist.get(2,0)}")
    
    class_weights = 1.0 / (counts + 1e-8)
    class_weights = torch.FloatTensor(class_weights / class_weights.sum() * 3.0) 
    print(f"      Applied Weights: {class_weights.tolist()}")

    # 85% Train, 15% Validation split
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Using device: {device}")
    
    # Move validation tensors to device
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    class_weights = class_weights.to(device)
    
    model = PriceTransformer(input_size=10, d_model=128, nhead=8, num_layers=3).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 100 
    batch_size = 64
    patience = 15
    patience_counter = 0
    best_val_loss = float('inf')

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)
    save_path = "models/price_model.pth"

    print(f"[3/4] Training PriceTransformer (Train: {len(X_train)}, Val: {len(X_val)})...")
    
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        epoch_train_loss = 0
        permutation = torch.randperm(X_train_tensor.size()[0])
        
        for i in range(0, len(X_train_tensor), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_tensor[indices].to(device)
            batch_y = y_train_tensor[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item() * len(batch_X)
            
        avg_train_loss = epoch_train_loss / len(X_train_tensor)
        
        # 2. Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
        # FIX: Step the scheduler with the validation loss
        scheduler.step(val_loss)
            
        # 3. Early Stopping & Best Model Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            status_flag = " [NEW BEST - SAVED]"
        else:
            patience_counter += 1
            status_flag = f" [Patience: {patience_counter}/{patience}]"

        if (epoch + 1) % 5 == 0 or status_flag == " [NEW BEST - SAVED]":
             print(f"      Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}{status_flag}")

        # Trigger Early Stop
        if patience_counter >= patience:
            print(f"\n[!] EARLY STOPPING TRIGGERED at Epoch {epoch+1}.")
            print(f"    Validation loss hasn't improved for {patience} epochs.")
            break

    print(f"\n[4/4] Training Complete. Best model logic preserved in {save_path}")

if __name__ == "__main__":
    train()