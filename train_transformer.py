import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from src.data.data_manager import DataManager
from src.models.price_predictor import PriceTransformer
from src.models.dataset import create_sequences

def train():
    print("[1/5] Fetching Multi-Timeframe Market Data (Including L2 Depth Synthesis)...")
    dm = DataManager()
    
    # Target timeframe
    df_1h = dm.get_crypto_data("kucoin", "BTC/USDT", "1h", 6000)
    # Peripheral vision timeframe
    df_4h = dm.get_crypto_data("kucoin", "BTC/USDT", "4h", 2000)
    
    if df_1h.empty or df_4h.empty:
        print("[ERROR] Failed to fetch sufficient data. Exiting.")
        return

    # Prepare 4h data for merging
    df_4h = df_4h.add_suffix('_4h')
    df_4h = df_4h.rename(columns={'timestamp_4h': 'timestamp'})
    
    # Merge on timestamp using forward-fill for the 4h context
    df_merged = pd.merge_asof(
        df_1h.sort_values('timestamp'),
        df_4h.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    df_merged.dropna(inplace=True)
    print(f"[INFO] Synced {len(df_merged)} bars across timeframes.")

    # Drop timestamp before sending to the model, leaving 56 distinct features
    data_values = df_merged.drop(columns=['timestamp']).values 
    
    # CRITICAL LEAKAGE FIX: Calculate mean/std ONLY on the training split (first 85%)
    # This prevents the model from "peeking" at the validation variance.
    split_row = int(len(data_values) * 0.85)
    train_slice = data_values[:split_row]
    
    data_mean = np.mean(train_slice, axis=0)
    data_std = np.std(train_slice, axis=0) + 1e-8
    normalized_data = (data_values - data_mean) / data_std

    print("[2/5] Building Volatility-Scaled Sequences (Seq Length: 50)...")
    
    X, y = create_sequences(normalized_data, seq_len=50)
    
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
    
    # Initialize model with updated configuration (num_layers=2)
    model = PriceTransformer(input_size=56, d_model=256, nhead=8, num_layers=2, dropout=0.5).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # LEARNING RATE ADJUSTMENT: Dropped to 5e-5 for stable Transformer training
    optimizer = optim.AdamW(model.parameters(), lr=5e-05, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 100 
    batch_size = 64
    
    # PATIENCE REDUCED: 8 epochs without improvement triggers termination
    patience = 8
    patience_counter = 0
    best_val_loss = float('inf')

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)
    save_path = "models/BTC_price_model.pth"

    print(f"[3/5] Training Diamond-Tier Transformer (Train: {len(X_train)}, Val: {len(X_val)})...")
    
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

    print(f"\n[4/5] Training Complete. Best model logic preserved in {save_path}")

    # [5/5] Full-Precision ONNX Export (No Quantization)
    print("\n[5/5] Exporting Full-Precision ONNX Model...")
    model.eval()
    model.to('cpu') 
    
    dummy_input = torch.randn(1, 50, 56)
    onnx_path = "models/BTC_price_model.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=18, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Success: {onnx_path} saved at full float32 precision.")

if __name__ == "__main__":
    train()