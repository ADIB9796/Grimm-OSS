import numpy as np

def create_sequences(data, seq_len=50):
    """
    Converts a normalized 2D array into 3D sequences for the Transformer.
    Implements Volatility-Scaled Labeling and auto-ingests L2 Depth arrays.
    
    Labels: 
    0: Down (Price drops > Dynamic Threshold)
    1: Neutral (Price stays within bounds)
    2: Up (Price rises > Dynamic Threshold)
    """
    X, y = [], []
    
    # Feature index 3 is 'close' price (Safe because L2 features are appended at the end)
    close_prices = data[:, 3]
    
    # Calculate percentage returns for volatility checking
    returns = np.diff(close_prices) / (np.abs(close_prices[:-1]) + 1e-8)
    
    vol_window = 100
    start_idx = max(0, vol_window - seq_len)

    for i in range(start_idx, len(data) - seq_len):
        # Window naturally captures all 56 features, including the new L2 arrays
        window = data[i : i + seq_len]
        
        current_idx = i + seq_len - 1
        next_idx = i + seq_len
        
        # Look at the most recent 100 bars of market returns
        recent_returns = returns[current_idx - vol_window : current_idx]
        
        if len(recent_returns) < vol_window:
            continue
            
        dynamic_threshold = np.std(recent_returns) * 2.5
        
        current_price = close_prices[current_idx]
        next_price = close_prices[next_idx]
        
        pct_change = (next_price - current_price) / (abs(current_price) + 1e-8)
        
        if pct_change > dynamic_threshold:
            label = 2  # UP
        elif pct_change < -dynamic_threshold:
            label = 0  # DOWN
        else:
            label = 1  # NEUTRAL
            
        X.append(window)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)