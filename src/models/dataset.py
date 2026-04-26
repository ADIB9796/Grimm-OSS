import numpy as np

def create_sequences(data, seq_len=50, threshold=0.002):
    """
    Converts a normalized 2D array into 3D sequences for the Transformer.
    Ensures we only use the first 10 features (OHLCV + Alpha Indicators).
    
    Labels: 
    0: Down (Price drops > threshold)
    1: Neutral (Price stays within bounds)
    2: Up (Price rises > threshold)
    """
    X, y = [], []
    data_10_features = data[:, :10] 
    
    # We stop early enough so we have a "next" candle to predict
    for i in range(len(data_10_features) - seq_len):
        window = data_10_features[i : (i + seq_len)]
        
        # Feature index 3 is 'close' price
        current_price = data_10_features[i + seq_len - 1, 3] 
        next_price = data_10_features[i + seq_len, 3]
        
        # Calculate true percentage change
        pct_change = (next_price - current_price) / (abs(current_price) + 1e-8)
        
        if pct_change > threshold:
            label = 2  # UP
        elif pct_change < -threshold:
            label = 0  # DOWN
        else:
            label = 1  # NEUTRAL
            
        X.append(window)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)