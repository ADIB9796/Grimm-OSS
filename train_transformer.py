import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.data_manager import DataManager
from src.models.price_predictor import PriceTransformer

def create_sequences(data, seq_length=50):
    sequences = []
    labels = []
    # SLICE: Ensure we only use the first 10 columns (the primary features)
    data_10_features = data[:, :10] 
    
    for i in range(len(data_10_features) - seq_length - 1):
        seq = data_10_features[i:(i + seq_length)]
        
        # Current vs Next Price for Classification
        current_price = data_10_features[i + seq_length - 1, 3] # Close price
        next_price = data_10_features[i + seq_length, 3]
        
        # VERSION 2.1 THRESHOLD HARDENING
        # Increasing to 0.3% to avoid fitting to market noise
        if next_price > current_price * 1.003:
            label = 1 # UP
        elif next_price < current_price * 0.997:
            label = 2 # DOWN
        else:
            label = 0 # NEUTRAL
            
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def train():
    print("[1/4] Fetching Market Data...")
    dm = DataManager()
    df = dm.get_crypto_data("kraken", "BTC/USD", "1h", 3000)
    
    # Normalize
    data_values = df.values
    data_mean = np.mean(data_values, axis=0)
    data_std = np.std(data_values, axis=0) + 1e-8
    normalized_data = (data_values - data_mean) / data_std

    print("[2/4] Building Sequences (Seq Length: 50)...")
    X, y = create_sequences(normalized_data)
    
    # DIAGNOSTIC: Check if we have enough UP/DOWN labels
    unique, counts = np.unique(y, return_counts=True)
    print(f"      Label Distribution: {dict(zip(unique, counts))}")
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Using device: {device}")
    
    # Model inherent expects 10 features
    model = PriceTransformer(input_size=10).to(device)
    
    # OPTIONAL: Weighting classes if distribution is skewed
    # If Neutral (0) is massive, we weight UP(1) and DOWN(2) higher
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lowered LR for stability

    epochs = 30 
    batch_size = 64

    print(f"[3/4] Training Transformer Model for {epochs} Epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size].to(device)
            batch_y = y_tensor[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"      Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(X_tensor)*batch_size:.4f}")

    torch.save(model.state_dict(), "models/price_model.pth")
    print("[4/4] Training Complete. Model saved to models/price_model.pth")

if __name__ == "__main__":
    train()