from src.risk.risk_manager import RiskManager


def main():
    print("Testing Risk Manager...\n")

    risk = RiskManager(balance=10000, risk_per_trade=0.02)

    entry_price = 50000
    stop_loss = 49000

    position_size = risk.calculate_position_size(entry_price, stop_loss)
    take_profit = risk.calculate_take_profit(entry_price)

    print(f"Position Size: {position_size:.4f}")
    print(f"Take Profit: {take_profit:.2f}")

    # Simulate profit
    risk.update_balance(10500)
    print(f"Updated Balance: {risk.balance}")
    print(f"Drawdown: {risk.current_drawdown():.2%}")
    print(f"Can Trade: {risk.can_trade()}")


if __name__ == "__main__":
    main()