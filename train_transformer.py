import os
import torch
import torch.nn as nn
from src.data.data_manager import DataManager
from src.models.price_predictor import PriceTransformer
from src.models.dataset import create_sequences

def train_model():
    print("[1/4] Fetching Market Data...")
    manager = DataManager()
    # Fetching the data
    data = manager.get_crypto_data(symbol="BTC/USD", exchange="kraken", timeframe="1h", limit=3000)
    
    # --- FIX: Ensure we are using the 10-feature set ---
    # Assuming your indicators are already in the dataframe returned by manager.get_crypto_data
    # If not, add your indicator calculation here (e.g., data = manager.add_technical_indicators(data))
    numeric_data = data.select_dtypes(include=[float, int]).copy()
    
    # Normalize features
    for col in numeric_data.columns:
        numeric_data[col] = (numeric_data[col] - numeric_data[col].mean()) / (numeric_data[col].std() + 1e-8)

    print(f"[2/4] Building Sequences (Seq Length: 50)...")
    X, y = create_sequences(numeric_data, seq_len=50, threshold=0.002)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Using device: {device}")

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X).to(device)
    y = torch.LongTensor(y).to(device)

    # --- FIX: Explicitly set input_size based on X ---
    input_size = X.shape[2] 
    print(f"[INFO] Training Transformer with {input_size} features...")
    
    model = PriceTransformer(input_size=input_size).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    print("\n[3/4] Training Transformer Model...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"      Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f}")

    # Save the weights
    os.makedirs("models", exist_ok=True)
    save_path = "models/price_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[4/4] Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_model()