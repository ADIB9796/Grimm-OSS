from src.data.data_manager import DataManager

manager = DataManager()

print("\nFetching Stock Data...")
print(manager.get_yfinance_data("AAPL", "2023-01-01", "2024-01-01").head())

print("\nFetching Crypto Data...")
print(manager.get_crypto_data("BTC/USDT").head())

# Uncomment after MetaTrader 5 is connected
# print("\nFetching Forex Data from MT5...")
# print(manager.get_mt5_data("EURUSD").head())