import numpy as np
import pandas as pd

def run_backtest(returns, signals, slippage=0.0005, position_size=1.0):
    """
    Simulates strategy performance with institutional slippage models and risk attribution.
    Utilizes T+1 execution logic and standardized downside risk metrics.
    """
    if len(returns) != len(signals):
        raise ValueError("Dimensional mismatch: returns and signals must align.")

    # Execution shift: generated today, executed at next open (T+1)
    # This prevents lookahead bias inherent in signal-to-close alignment.
    execution_signals = np.roll(signals, 1) * float(position_size)
    execution_signals[0] = 0

    daily_results = execution_signals * returns

    # Linear transaction costs (Basis Points)
    position_changes = np.abs(np.diff(execution_signals, prepend=0))
    daily_results -= position_changes * slippage

    # Cumulative Performance
    profit_path = np.cumprod(1 + daily_results) - 1
    market_path = np.cumprod(1 + returns) - 1

    # Annualization (252-day standard)
    total_years = len(returns) / 252.0
    if total_years > 0:
        annual_growth = (1 + profit_path[-1]) ** (1 / total_years) - 1
    else:
        annual_growth = 0

    # Risk Attribution: Sharpe
    volatility = np.std(daily_results) * np.sqrt(252)
    sharpe = annual_growth / volatility if volatility > 1e-9 else 0

    # Risk Attribution: Sortino (Standard Downside Deviation)
    # Correct formula: Root Mean Square of negative returns relative to 0.
    neg_rets = daily_results[daily_results < 0]
    if len(neg_rets) > 0:
        # Prevent division by zero if negative returns are identical or infinitesimal
        downside_var = np.mean(neg_rets**2)
        downside_deviation = np.sqrt(downside_var) * np.sqrt(252)
        sortino = annual_growth / max(1e-9, downside_deviation)
    else:
        # Zero-loss scenario: sortino is undefined; use sharpe as a reasonable proxy
        sortino = sharpe

    # Risk Attribution: Calmar (Drawdown Efficiency)
    peak = np.maximum.accumulate(1 + profit_path)
    drawdowns = (1 + profit_path) / peak - 1
    max_dd = np.min(drawdowns)
    calmar = annual_growth / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    # Profit Factor & Expectancy
    gains = daily_results[daily_results > 0]
    losses = daily_results[daily_results < 0]
    
    gross_gains  = np.sum(gains)
    gross_losses = abs(np.sum(losses))
    profit_factor = gross_gains / gross_losses if gross_losses > 1e-9 else (1.0 if gross_gains == 0 else 10.0)

    active_days = np.sum(execution_signals != 0)
    win_rate = np.sum(daily_results > 0) / active_days if active_days > 0 else 0
    
    avg_win = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Count transaction events (entries + reversals)
    total_trades = int(np.sum(np.diff(execution_signals, prepend=0) != 0))

    return {
        "final_profit": profit_path[-1],
        "annual_return": annual_growth,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": total_trades, # Required for intelligence engine
        "max_win": np.max(daily_results) if len(daily_results) > 0 else 0,
        "max_loss": np.min(daily_results) if len(daily_results) > 0 else 0,
        "equity_curve": profit_path,
        "market_curve": market_path,
        "drawdown_curve": drawdowns,
        "daily_returns": daily_results
    }


def compute_kelly_fraction(returns, signals, half=True):
    """
    Optimal position sizing via the Kelly Criterion (Shifted for T+1 execution).
    f* = mu / sigma^2
    """
    clean_returns = np.nan_to_num(returns)
    execution_signals = np.roll(signals, 1)
    execution_signals[0] = 0
    strategy_returns = execution_signals * clean_returns

    active = strategy_returns[strategy_returns != 0]
    if len(active) < 20: 
        return 0.0

    mu = np.mean(active)
    sigma2 = np.var(active)
    if sigma2 < 1e-12 or mu <= 0:
        return 0.0

    kelly = mu / sigma2
    if half:
        kelly /= 2.0

    # Safety Guard: Professional capital management requires non-negative 
    # position sizes and limits to prevent over-leverage in high-stability windows.
    # T+1 logic ensures sizing is based on signal availability at trading time.
    kelly_constrained = float(np.clip(kelly, 0.05, 1.0))

    return kelly_constrained


def coherence_stability(coherence_grid, band_idx, window=50):
    """
    Measures the temporal stability of cross-asset resonance.
    High stability suggests a reliable predictive relationship.
    """
    band = coherence_grid[band_idx, :]
    n = len(band)
    result = np.full(n, np.nan)
    for t in range(window, n):
        result[t] = np.std(band[t - window : t])
    return result


def create_signals_from_resonance(coherence_grid, band_idx, high_barrier=0.7, low_barrier=0.3):
    """Generates signals based on absolute resonance thresholds."""
    resonance = coherence_grid[band_idx, :]
    actions = np.zeros_like(resonance)
    actions[resonance < low_barrier] = 1
    actions[resonance > high_barrier] = -1
    return actions


def create_phase_signals(coherence_grid, phase_grid, source_returns, band_idx,
                          coherence_threshold=0.6, min_phase_strength=0.3, smoothing=5):
    """
    Directional signals derived from cross-asset phase relationships.
    Trades the 'Lead/Lag' offset by following the source asset's recent momentum.
    """
    coherence = coherence_grid[band_idx, :]
    phase = phase_grid[band_idx, :]
    n = len(coherence)
    signals = np.zeros(n)

    for t in range(smoothing + 1, n):
        if coherence[t] < coherence_threshold:
            continue

        # Weighted Lead detection
        avg_phase = np.mean(phase[t - smoothing : t])
        if avg_phase > -min_phase_strength:
            continue

        # Directional following
        recent_source = np.sum(source_returns[max(0, t - smoothing) : t])
        signals[t] = 1.0 if recent_source > 0 else -1.0

    return signals


def apply_trend_filter(signals, prices, ma_period=50):
    """Macro regime filter: inhibits signals that oppose the primary structural trend."""
    prices = np.asarray(prices, dtype=float)
    filtered = np.array(signals, dtype=float)

    cumsum = np.nancumsum(prices)
    ma = np.full_like(prices, np.nan)
    ma[ma_period - 1:] = (cumsum[ma_period - 1:] - np.concatenate([[0], cumsum[:-ma_period]])) / ma_period

    filtered[(filtered == 1)  & (prices < ma)] = 0
    filtered[(filtered == -1) & (prices > ma)] = 0

    return filtered
