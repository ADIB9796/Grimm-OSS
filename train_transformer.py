import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.data_manager import DataManager
from src.models.price_predictor import PriceTransformer
from src.data.dataset import create_sequences

def train():
    print("[1/4] Fetching Market Data...")
    dm = DataManager()
    
    # FIX 1: Switched from Kraken to Binance for deep historical access
    df = dm.get_crypto_data("binance", "BTC/USDT", "1h", 3500)
    
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
    
    # FIX 2: Increased threshold to 0.5% to bring back the "Neutral" class
    X, y = create_sequences(normalized_data, seq_len=50, threshold=0.005)
    
    # Calculate Class Weights
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"      Label Distribution: Down(0): {dist.get(0,0)}, Neutral(1): {dist.get(1,0)}, Up(2): {dist.get(2,0)}")
    
    class_weights = 1.0 / (counts + 1e-8)
    class_weights = torch.FloatTensor(class_weights / class_weights.sum() * 3.0) 
    print(f"      Applied Weights: {class_weights.tolist()}")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Using device: {device}")
    class_weights = class_weights.to(device)
    
    # Updated hyperparams to match the new robust predictor
    model = PriceTransformer(input_size=10, d_model=128, nhead=8, num_layers=3).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    epochs = 40 
    batch_size = 64

    print(f"[3/4] Training PriceTransformer on {len(X)} samples...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        permutation = torch.randperm(X_tensor.size()[0])
        
        for i in range(0, len(X_tensor), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_tensor[indices].to(device)
            batch_y = y_tensor[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)
            
        scheduler.step()
        avg_loss = epoch_loss / len(X_tensor)
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "models/price_model.pth")
    print("[4/4] Training Complete. Model saved to models/price_model.pth")

if __name__ == "__main__":
    train()