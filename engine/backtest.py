import numpy as np
import pandas as pd


def run_backtest(returns, signals, slippage=0.0005, position_size=1.0):
    """
    Simulates performance with institutional risk metrics.
    Assumes a 1-day execution lag to prevent lookahead bias.
    """
    if len(returns) != len(signals):
        raise ValueError("Data mismatch: returns and signals aren't the same length.")

    # Execution shift: generated today, executed tomorrow
    execution_signals = np.roll(signals, 1) * float(position_size)
    execution_signals[0] = 0

    daily_results = execution_signals * returns

    # Flat transaction cost on every change in position
    position_changes = np.abs(np.diff(execution_signals, prepend=0))
    daily_results -= position_changes * slippage

    # Equity curves as cumulative growth of $1
    profit_path = np.cumprod(1 + daily_results) - 1
    market_path = np.cumprod(1 + returns) - 1

    # Win rate and Profit Factor
    gains = daily_results[daily_results > 0]
    losses = daily_results[daily_results < 0]
    
    win_rate = len(gains) / (len(gains) + len(losses)) if (len(gains) + len(losses)) > 0 else 0
    profit_factor = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else (1.0 if len(gains) > 0 else 0)

    total_years = len(returns) / 252.0
    if total_years > 0:
        annual_growth = (1 + profit_path[-1]) ** (1 / total_years) - 1
        market_growth = (1 + market_path[-1]) ** (1 / total_years) - 1
    else:
        annual_growth, market_growth = 0, 0

    # Sharp / Sortino
    volatility = np.std(daily_results) * np.sqrt(252)
    sharpe = annual_growth / volatility if volatility > 0 else 0
    
    downside_vol = np.std(daily_results[daily_results < 0]) * np.sqrt(252) if len(daily_results[daily_results < 0]) > 2 else volatility
    sortino = annual_growth / downside_vol if downside_vol > 0 else 0

    # Max Drawdown
    peak = np.maximum.accumulate(1 + profit_path)
    drawdowns = (1 + profit_path) / peak - 1
    worst_drawdown = np.min(drawdowns)
    
    # Calmar
    calmar = annual_growth / abs(worst_drawdown) if worst_drawdown != 0 else 0

    return {
        "final_profit": profit_path[-1],
        "annual_return": annual_growth,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "max_drawdown": worst_drawdown,
        "drawdown_curve": drawdowns,
        "win_rate": win_rate,
        "equity_curve": profit_path,
        "market_curve": market_path,
        "daily_returns": daily_results,
        "position_size": position_size,
    }


def compute_kelly_fraction(returns, signals, half=True):
    """
    Optimal position sizing via the Kelly Criterion.
    """
    shifted = np.roll(signals, 1)
    shifted[0] = 0
    strategy_returns = shifted * returns

    active = strategy_returns[strategy_returns != 0]
    if len(active) < 15:
        return 0.0

    mu = np.mean(active)
    sigma2 = np.var(active)

    if sigma2 < 1e-12 or mu <= 0:
        return 0.0

    kelly = mu / sigma2
    if half:
        kelly /= 2.0

    return float(np.clip(kelly, 0.05, 1.0))


def coherence_stability(coherence_grid, band_idx, window=50):
    """
    Rolling standard deviation of coherence at a given frequency band.
    """
    band = coherence_grid[band_idx, :]
    n = len(band)
    result = np.full(n, np.nan)
    for t in range(window, n):
        result[t] = np.std(band[t - window : t])
    return result


def create_signals_from_resonance(coherence_grid, band_idx,
                                   high_barrier=0.7, low_barrier=0.3):
    """
    Mean-reversion logic based on coherence level.
    """
    resonance = coherence_grid[band_idx, :]
    actions = np.zeros_like(resonance)
    actions[resonance < low_barrier] = 1
    actions[resonance > high_barrier] = -1
    return actions


def create_phase_signals(coherence_grid, phase_grid, source_returns, band_idx, freqs,
                          coherence_threshold=0.6, min_phase_strength=0.3,
                          smoothing=5):
    """
    Directional signals derived from lead-adjusted wavelet phase angle.
    
    This version uses the phase-derived 'Lead Time' to look back at the 
    corresponding historical shift in the source asset, providing a 
    much more robust directional hint than simple trend-following.
    """
    coherence = coherence_grid[band_idx, :]
    phase = phase_grid[band_idx, :]
    period = 1.0 / freqs[band_idx] if freqs[band_idx] > 0 else 0
    
    n = len(coherence)
    signals = np.zeros(n)
    
    current_signal = 0

    for t in range(max(20, smoothing + 1), n):
        # 1. Check relationship strength
        if coherence[t] < coherence_threshold:
            current_signal *= 0.8  # Decay signal if relationship breaks
            if abs(current_signal) < 0.1: current_signal = 0
            signals[t] = np.sign(current_signal) if abs(current_signal) > 0.5 else 0
            continue 

        # 2. Extract Lead Time (L)
        avg_phase = np.mean(phase[t - smoothing : t])
        
        # Only act when source is clearly leading (phase clearly negative in pycwt convention)
        if avg_phase > -min_phase_strength:
            signals[t] = 0
            continue

        # L = (Phase * Period) / 2pi. 
        # If phase = -pi/2 and period = 20, L = -5 days.
        lead_bars = int(round(avg_phase * period / (2 * np.pi)))
        
        # 3. Synchronized Directional Check (Momentum-Weighted Anchor)
        # To predict Traded(t+1), we shift to the source asset's 'preview' window.
        # Theoretical Anchor: t_traded = t_source + lead_bars
        lookback_idx = t + 1 + lead_bars
        if lookback_idx < 1: 
            signals[t] = 0
            continue
            
        # Refined Directional Trigger: 
        # Instead of a simple mean, we anchor on the instantaneous momentum at the 
        # lead-offset and confirm it with a 3-day 'mini-trend'.
        anchor_momentum = source_returns[lookback_idx] 
        mini_trend      = np.sum(source_returns[max(0, lookback_idx - 2) : lookback_idx + 1])
        
        # High-confidence signal: Anchor and mini-trend must align
        if anchor_momentum > 0 and mini_trend > 0:
            current_signal = 1
        elif anchor_momentum < 0 and mini_trend < 0:
            current_signal = -1
        else:
            # Low confidence: Maintain current position but don't flip
            # This handles 'noisy' leadership signals at the crest of a cycle.
            pass
        
        signals[t] = current_signal

    return signals


def apply_trend_filter(signals, prices, ma_period=50):
    """
    Macro regime filter.
    """
    prices = np.asarray(prices, dtype=float)
    filtered = np.array(signals, dtype=float)

    cumsum = np.nancumsum(prices)
    ma = np.full_like(prices, np.nan)
    ma[ma_period - 1:] = (
        cumsum[ma_period - 1:]
        - np.concatenate([[0], cumsum[:-ma_period]])
    ) / ma_period

    filtered[(filtered == 1)  & (prices < ma)] = 0
    filtered[(filtered == -1) & (prices > ma)] = 0

    return filtered
