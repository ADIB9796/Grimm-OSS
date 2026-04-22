from src.data.data_manager import DataManager
from src.strategies import (
    MovingAverageStrategy,
    RSIStrategy
)


def main():
    manager = DataManager()

    # Fetch crypto data using CCXT
    data = manager.get_crypto_data(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        limit=500
    )

    print("\nMarket Data:")
    print(data.tail())

    # Moving Average Strategy
    ma_strategy = MovingAverageStrategy()
    ma_results = ma_strategy.generate_signals(data)

    print("\nMoving Average Signals:")
    print(ma_results[["timestamp", "close", "signal"]].tail())

    # RSI Strategy
    rsi_strategy = RSIStrategy()
    rsi_results = rsi_strategy.generate_signals(data)

    print("\nRSI Signals:")
    print(rsi_results[["timestamp", "close", "rsi", "signal"]].tail())


if __name__ == "__main__":
    main()