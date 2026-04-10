import numpy as np
import pandas as pd


def run_backtest(returns, signals, slippage=0.0005, position_size=1.0):
    """
    Simulates how a trading strategy would have performed over time.
    We assume a 1-day lag for signals to make sure there is no lookahead bias.

    position_size: fraction of capital to deploy on each trade (0.0–1.0).
                   Use compute_kelly_fraction() to calculate an optimal value.
    """
    if len(returns) != len(signals):
        raise ValueError("Data mismatch: returns and signals aren't the same length.")

    # Execution shift: generated today, executed tomorrow
    execution_signals = np.roll(signals, 1) * float(position_size)
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
        "daily_returns": daily_results,
        "position_size": position_size,
    }


def compute_kelly_fraction(returns, signals, half=True):
    """
    Estimates the optimal position size using the Kelly Criterion.

    Uses the continuous-return formula: f* = μ / σ²  (mean over variance)
    applied to the strategy returns (signal × market return, un-sized).

    half=True uses half-Kelly, which is the practical standard — it cuts the
    theoretical maximum by 50 % to account for estimation error and reduce
    the risk of ruin in adverse sequences.

    Returns a fraction in [0, 1].  Falls back to 1.0 when there is
    insufficient data or the strategy has no positive expectancy.
    """
    # Preview strategy returns at full size (no position scaling)
    shifted = np.roll(signals, 1)
    shifted[0] = 0
    strategy_returns = shifted * returns

    active = strategy_returns[strategy_returns != 0]
    if len(active) < 20:
        return 1.0

    mu = np.mean(active)
    sigma2 = np.var(active)

    if sigma2 < 1e-12 or mu <= 0:
        return 1.0  # no positive edge — use full size (no leverage effect)

    kelly = mu / sigma2
    if half:
        kelly /= 2.0

    return float(np.clip(kelly, 0.05, 1.0))


def apply_trend_filter(signals, prices, ma_period=50):
    """
    Prevents the strategy from fighting the macro trend.

    - Long signals (+1) are kept only when price is above its moving average.
    - Short signals (-1) are kept only when price is below its moving average.
    - Signals that go against the trend are zeroed out.

    This avoids shorting into strong up-moves and going long into breakdowns —
    a common failure mode of pure mean-reversion strategies.
    """
    prices = np.asarray(prices, dtype=float)
    filtered = np.array(signals, dtype=float)

    # Rolling mean via cumsum — O(n), no loop
    cumsum = np.nancumsum(prices)
    ma = np.full_like(prices, np.nan)
    ma[ma_period - 1:] = (cumsum[ma_period - 1:] - np.concatenate([[0], cumsum[:-ma_period]])) / ma_period

    uptrend = prices > ma      # True where price > MA (long-friendly)
    downtrend = prices < ma    # True where price < MA (short-friendly)

    # Zero out signals that contradict the trend
    filtered[(filtered == 1) & ~uptrend] = 0
    filtered[(filtered == -1) & ~downtrend] = 0

    return filtered


def create_signals_from_resonance(coherence_grid, band_idx, high_barrier=0.7, low_barrier=0.3):
    """
    A simple "Mean Reversion" logic based on asset resonance.
    - If they decouple too much (low resonance), we expect them to snap back → Long.
    - If they are overly coupled, we step aside or bet against the consensus → Short.
    """
    resonance_levels = coherence_grid[band_idx, :]
    actions = np.zeros_like(resonance_levels)

    actions[resonance_levels < low_barrier] = 1
    actions[resonance_levels > high_barrier] = -1

    return actions
