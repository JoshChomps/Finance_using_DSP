import numpy as np
import pandas as pd

def run_backtest(returns, signals, slippage=0.0005):
    """
    Simulates how a trading strategy would have performed over time.
    We assume a 1-day lag for signals to make sure there is no lookahead bias.
    """
    if len(returns) != len(signals):
        raise ValueError("Data mismatch: returns and signals aren't the same length.")

    # Execution shift: generated today, executed tomorrow
    execution_signals = np.roll(signals, 1)
    execution_signals[0] = 0
    
    # Calculate daily strategy performance
    daily_results = execution_signals * returns
    
    # Simple transaction cost model on every trade change
    position_changes = np.abs(np.diff(execution_signals, prepend=0))
    daily_results -= position_changes * slippage
    
    # Track the growth of $1 (equity curves)
    profit_path = np.cumprod(1 + daily_results) - 1
    market_path = np.cumprod(1 + returns) - 1
    
    # Basic stats
    profitable_days = np.sum(daily_results > 0)
    active_days = np.sum(daily_results != 0)
    win_rate = profitable_days / active_days if active_days > 0 else 0
    
    # Annualized projections (assuming 252 trading days)
    total_days = len(returns)
    total_years = total_days / 252.0
    
    if total_years > 0:
        annual_growth = (1 + profit_path[-1]) ** (1 / total_years) - 1
        market_growth = (1 + market_path[-1]) ** (1 / total_years) - 1
    else:
        annual_growth, market_growth = 0, 0
        
    volatility = np.std(daily_results) * np.sqrt(252)
    sharpe = annual_growth / volatility if volatility > 0 else 0
    
    # Find the deepest valley (Maximum Drawdown)
    peak = np.maximum.accumulate(1 + profit_path)
    drawdowns = (1 + profit_path) / peak - 1
    worst_drawdown = np.min(drawdowns)
    
    return {
        "final_profit": profit_path[-1],
        "annual_return": annual_growth,
        "sharpe": sharpe,
        "max_drawdown": worst_drawdown,
        "win_rate": win_rate,
        "equity_curve": profit_path,
        "market_curve": market_path,
        "daily_returns": daily_results
    }

def create_signals_from_resonance(coherence_grid, band_idx, high_barrier=0.7, low_barrier=0.3):
    """
    A simple "Mean Reversion" logic based on asset resonance.
    - If they decouple too much (low resonance), we expect them to snap back.
    - If they are overly coupled, we step aside or bet against the consensus.
    """
    resonance_levels = coherence_grid[band_idx, :]
    actions = np.zeros_like(resonance_levels)
    
    # Long signal on decoupling
    actions[resonance_levels < low_barrier] = 1 
    
    # Short/Neutral signal on high coupling
    actions[resonance_levels > high_barrier] = -1
    
    return actions

