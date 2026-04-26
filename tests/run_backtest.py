import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.data_manager import DataManager
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent

# =========================
# DATA LOADER
# =========================
def load_backtest_data(symbol="BTC/USD", limit=500):
    """
    Fetches fresh Out-Of-Sample data for backtesting using our robust DataManager.
    You can replace this with a pd.read_csv if you want to test a specific static file.
    """
    print(f"[INFO] Fetching {limit} recent bars for {symbol} backtest...")
    dm = DataManager()
    df = dm.get_crypto_data("kraken", symbol, "1h", limit=limit)
    
    if df.empty:
        raise ValueError("No data fetched. Check internet connection or exchange limits.")
        
    return df

# =========================
# LOAD AGENT
# =========================
def load_trained_agent(env):
    """
    Loads the trained RL Agent. 
    Note: PriceTransformer is already loaded internally by the TradingEnvironment.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # We pass standard params, but the loaded weights will dictate behavior
    agent = RLAgent(state_size, action_size)
    
    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading RL Agent weights to {device}...")
    agent.model.load_state_dict(torch.load("models/rl_trading_model.pth", map_location=device, weights_only=True))
    
    # Force pure exploitation (no random guessing during backtest)
    agent.epsilon = 0.0 
    agent.model.eval()
    
    return agent

# =========================
# BACKTEST ENGINE
# =========================
def run_backtest(env, agent):
    print("[INFO] Starting Backtest Simulation...")
    state, _ = env.reset()
    
    prices = []
    equity = []
    actions = []
    
    done = False
    truncated = False
    
    while not (done or truncated):
        # 1. Agent Decides
        action = agent.act(state)
        
        # 2. Environment Steps (Gymnasium 5-tuple standard)
        next_state, reward, done, truncated, info = env.step(action)
        
        # 3. Track Metrics
        # Current price can be extracted from the raw data using the env's internal step tracker
        current_idx = env.current_step + (len(env.raw_data) - len(env.data)) - 1
        current_price = env.raw_data.iloc[current_idx]["close"]
        
        prices.append(current_price)
        equity.append(env.risk_manager.balance)
        actions.append(action)
        
        state = next_state

    print("[INFO] Backtest Complete.")
    return prices, equity, actions

# =========================
# METRICS
# =========================
def compute_metrics(equity, initial_balance):
    equity_array = np.array(equity)
    
    # Returns Profile
    pct_returns = np.diff(equity_array) / equity_array[:-1]
    
    # Sharpe Ratio (Simplified Annualized, assuming 1h candles -> 8760/yr)
    mean_ret = np.mean(pct_returns) if len(pct_returns) > 0 else 0
    std_ret = np.std(pct_returns) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(8760) 
    
    # Maximum Drawdown (True Peak-to-Trough percentage)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_dd = np.max(drawdown)

    # Win Rate & Profit Factor (Trade-based approximation from equity bumps)
    wins = pct_returns[pct_returns > 0]
    losses = pct_returns[pct_returns < 0]
    
    win_rate = len(wins) / len(pct_returns) if len(pct_returns) > 0 else 0
    gross_profit = np.sum(wins)
    gross_loss = np.abs(np.sum(losses))
    profit_factor = gross_profit / (gross_loss + 1e-8) if gross_loss > 0 else float('inf')
    
    total_return = ((equity[-1] - initial_balance) / initial_balance) * 100

    return sharpe, max_dd, win_rate, profit_factor, total_return

# =========================
# PLOT DASHBOARD
# =========================
def plot_dashboard(df, prices, equity, actions):
    print("[INFO] Generating Plotly Dashboard...")
    
    # Align dataframe index with the tracked arrays
    # (The environment trims the first few rows for indicators)
    plot_df = df.iloc[-len(prices):].copy()
    timestamps = plot_df["timestamp"].dt.strftime('%Y-%m-%d %H:%M')

    # Create subplots: 2 Rows, Shared X-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Market Price & Execution", "Equity Curve")
    )

    # --- ROW 1: PRICE ---
    fig.add_trace(go.Scatter(
        x=timestamps, y=prices,
        name="Close Price",
        line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
    ), row=1, col=1)

    # Extract Buy/Sell coordinates
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    
    for i, a in enumerate(actions):
        if a == 1: # BUY
            buy_x.append(timestamps.iloc[i])
            buy_y.append(prices[i])
        elif a == 2: # SELL
            sell_x.append(timestamps.iloc[i])
            sell_y.append(prices[i])

    fig.add_trace(go.Scatter(
        x=buy_x, y=buy_y,
        mode="markers",
        name="Long Entry",
        marker=dict(symbol="triangle-up", size=12, color="springgreen", line=dict(width=1, color="black"))
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_x, y=sell_y,
        mode="markers",
        name="Close Position",
        marker=dict(symbol="triangle-down", size=12, color="crimson", line=dict(width=1, color="black"))
    ), row=1, col=1)

    # --- ROW 2: EQUITY ---
    fig.add_trace(go.Scatter(
        x=timestamps, y=equity,
        name="Portfolio Value",
        line=dict(color='cyan', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ), row=2, col=1)

    # Layout Enhancements
    fig.update_layout(
        title="Grimm-OSS: V2.2 Backtest Performance",
        template="plotly_dark",
        hovermode="x unified",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Capital ($)", row=2, col=1)

    fig.show()

# =========================
# MAIN ENTRY
# =========================
def main():
    # 1. Setup
    initial_capital = 10000.0
    df = load_backtest_data(symbol="BTC/USD", limit=720) # Approx 1 month of 1h data
    
    # 2. Initialize Environment
    env = TradingEnvironment(df, initial_balance=initial_capital)
    
    # 3. Load Agent
    agent = load_trained_agent(env)
    
    # 4. Run Simulation
    prices, equity, actions = run_backtest(env, agent)
    
    # 5. Report Metrics
    sharpe, dd, win_rate, pf, total_ret = compute_metrics(equity, initial_capital)
    
    print("\n" + "="*40)
    print("      BACKTEST PERFORMANCE")
    print("="*40)
    print(f" Initial Balance:  ${initial_capital:,.2f}")
    print(f" Final Balance:    ${equity[-1]:,.2f}")
    print(f" Total Return:     {total_ret:+.2f}%")
    print(f" Max Drawdown:     {dd*100:.2f}%")
    print(f" Win Rate:         {win_rate*100:.1f}%")
    print(f" Profit Factor:    {pf:.2f}")
    print(f" Est. Sharpe:      {sharpe:.2f}")
    print("="*40 + "\n")
    
    # 6. Visualize
    plot_dashboard(df, prices, equity, actions)

if __name__ == "__main__":
    main()