import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.data_manager import DataManager
from src.models.price_predictor import PriceTransformer

def create_sequences(data, seq_length=50):
    sequences = []
    labels = []
    data_10_features = data[:, :10] 
    
    for i in range(len(data_10_features) - seq_length - 1):
        seq = data_10_features[i:(i + seq_length)]
        
        current_price = data_10_features[i + seq_length - 1, 3] 
        next_price = data_10_features[i + seq_length, 3]
        
        # TWEAK: Relaxed threshold slightly to 0.2% to capture more 'Neutral' states
        # On 1h charts, 0.3% can be too aggressive for BTC during consolidation
        if next_price > current_price * 1.002:
            label = 1 # UP
        elif next_price < current_price * 0.998:
            label = 2 # DOWN
        else:
            label = 0 # NEUTRAL
            
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def train():
    dm = DataManager()
    # Fetching 3000 now works thanks to the loop in DataManager
    df = dm.get_crypto_data("kraken", "BTC/USD", "1h", 3000)
    
    data_values = df.values
    data_mean = np.mean(data_values, axis=0)
    data_std = np.std(data_values, axis=0) + 1e-8
    normalized_data = (data_values - data_mean) / data_std

    X, y = create_sequences(normalized_data)
    
    # CALCULATE CLASS WEIGHTS
    # This prevents the model from ignoring the minority class (Neutral)
    unique, counts = np.unique(y, return_counts=True)
    print(f"      Label Distribution: {dict(zip(unique, counts))}")
    
    class_weights = 1.0 / counts
    class_weights = torch.FloatTensor(class_weights / class_weights.sum() * 3.0) 
    print(f"      Applied Weights: {class_weights.tolist()}")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    
    model = PriceTransformer(input_size=10).to(device)
    
    # CRITICAL: Passing weights to the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0003) # Slower LR for better convergence

    epochs = 40 # Increased to allow weights to take effect
    batch_size = 64

    print(f"[3/4] Training PriceTransformer on {len(X)} samples...")
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
            
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(X_tensor)*batch_size:.4f}")

    torch.save(model.state_dict(), "models/price_model.pth")
    print("[4/4] Training Complete.")

if __name__ == "__main__":
    train()