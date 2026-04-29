import unittest
from src.execution.paper_trader import PaperTrader

class TestPaperTrader(unittest.TestCase):
    
    def setUp(self):
        # Exness Simulation: 0% fees, 0.15% spread, 10x leverage
        self.trader = PaperTrader(
            initial_balance=1000.0, 
            fee_rate=0.0, 
            slippage_pct=0.0015,
            leverage=10.0
        )
        
    def test_exness_buy_execution_and_spread(self):
        # Allocate $100 margin at 10x leverage = $1000 purchasing power
        # Price is 50,000, but spread adds 0.15% -> 50,075
        receipt = self.trader.execute_trade(action=1, current_price=50000.0, allocated_margin=100.0, timestamp="T1")
        
        self.assertEqual(receipt["status"], "FILLED_BUY")
        self.assertEqual(receipt["fees"], 0.0) # Zero commission
        
        # Balance drops by locked margin ($100)
        self.assertEqual(self.trader.current_balance, 900.0) 
        self.assertEqual(self.trader.margin_used, 100.0)
        
        # Executed price should include 0.15% spread
        self.assertEqual(receipt["executed_price"], 50075.0)
        
        # Asset size = $1000 / 50075 = 0.0199700...
        self.assertAlmostEqual(self.trader.position_size_asset, 1000 / 50075.0)

    def test_exness_sell_execution_and_leveraged_pnl(self):
        # Buy at $50,000
        self.trader.execute_trade(action=1, current_price=50000.0, allocated_margin=100.0, timestamp="T1")
        
        # Price pumps 10% to $55,000
        receipt = self.trader.execute_trade(action=2, current_price=55000.0, allocated_margin=0.0, timestamp="T2")
        
        self.assertEqual(receipt["status"], "FILLED_SELL")
        self.assertFalse(self.trader.is_in_position)
        
        # Check PnL Math
        # Entry Price: 50000 * 1.0015 = 50075
        # Exit Price: 55000 * 0.9985 = 54917.5
        # Amount BTC: 1000 / 50075 = 0.01997004
        # Gross Return: 0.01997004 * 54917.5 = 1096.70
        # Entry Value: 1000
        # PnL = 1096.70 - 1000 = +96.70
        
        self.assertTrue(receipt["pnl"] > 0)
        self.assertAlmostEqual(receipt["pnl"], 96.7049, places=2)
        
        # Final Balance = 900 (Free) + 100 (Margin Returned) + 96.70 (PnL)
        self.assertAlmostEqual(self.trader.current_balance, 1096.7049, places=2)

if __name__ == '__main__':
    unittest.main()