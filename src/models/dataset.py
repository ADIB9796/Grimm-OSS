import numpy as np

def create_sequences(df, seq_len=50, threshold=0.001):
    """
    Converts a 2D DataFrame into 3D sequences for the Transformer.
    Labels: 0 (Down), 1 (Neutral), 2 (Up).
    """
    X, y = [], []
    
    # Calculate future returns to create our labels (predicting the next candle)
    future_returns = df['close'].pct_change().shift(-1).fillna(0).values
    features = df.values

    for i in range(len(df) - seq_len):
        # Extract a window of 'seq_len' rows
        window = features[i : (i + seq_len)]
        X.append(window)
        
        # Determine the label based on the next candle's return
        ret = future_returns[i + seq_len - 1]
        
        if ret > threshold:
            y.append(2)  # Price goes up significantly
        elif ret < -threshold:
            y.append(0)  # Price goes down significantly
        else:
            y.append(1)  # Sideways / Neutral market

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)