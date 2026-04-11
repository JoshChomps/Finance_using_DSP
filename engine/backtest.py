import numpy as np
import pandas as pd


def run_backtest(returns, signals, slippage=0.0005, position_size=1.0):
    """
    Simulates how a trading strategy would have performed over time.
    Assumes a 1-day execution lag to prevent lookahead bias.

    position_size: fraction of capital to deploy per trade (0.0–1.0).
                   Use compute_kelly_fraction() to derive an optimal value.
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

    profitable_days = np.sum(daily_results > 0)
    active_days = np.sum(daily_results != 0)
    win_rate = profitable_days / active_days if active_days > 0 else 0

    total_years = len(returns) / 252.0
    if total_years > 0:
        annual_growth = (1 + profit_path[-1]) ** (1 / total_years) - 1
        market_growth = (1 + market_path[-1]) ** (1 / total_years) - 1
    else:
        annual_growth, market_growth = 0, 0

    volatility = np.std(daily_results) * np.sqrt(252)
    sharpe = annual_growth / volatility if volatility > 0 else 0

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
    Optimal position sizing via the Kelly Criterion.

    Formula: f* = μ / σ²  (continuous-return version)
    where μ and σ² are estimated from the strategy's un-sized daily returns.

    half=True (default) applies half-Kelly: 50% of the theoretical optimum.
    This accounts for estimation error in μ and σ² and meaningfully reduces
    drawdown depth without sacrificing much long-run growth rate.

    Returns a fraction in [0.05, 1.0], or 0.0 when the strategy has no
    positive expectancy (μ ≤ 0) — in that case no trade should be taken.
    """
    shifted = np.roll(signals, 1)
    shifted[0] = 0
    strategy_returns = shifted * returns

    active = strategy_returns[strategy_returns != 0]
    if len(active) < 20:
        # Insufficient history — treat as no-edge and stay out
        return 0.0

    mu = np.mean(active)
    sigma2 = np.var(active)

    if sigma2 < 1e-12 or mu <= 0:
        # No positive edge: Kelly says bet nothing
        return 0.0

    kelly = mu / sigma2
    if half:
        kelly /= 2.0

    return float(np.clip(kelly, 0.05, 1.0))


def coherence_stability(coherence_grid, band_idx, window=50):
    """
    Rolling standard deviation of coherence at a given frequency band.

    A low value means the relationship is stable and reliable.
    A high value means the coherence is erratic — signals will be noisy
    and the strategy is unlikely to have a consistent edge at this band.

    Returns an array of rolling std values (NaN for the first `window` bars).
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
    Mean-reversion logic based on coherence level alone.

    NOTE: This signal has no directional information — it bets on
    reversion without knowing which asset should move to converge.
    Use create_phase_signals() for a directionally-informed signal.

    - coherence < low_barrier  → +1 (long: expect re-coupling)
    - coherence > high_barrier → -1 (short: fade the over-coupling)
    """
    resonance = coherence_grid[band_idx, :]
    actions = np.zeros_like(resonance)
    actions[resonance < low_barrier] = 1
    actions[resonance > high_barrier] = -1
    return actions


def create_phase_signals(coherence_grid, phase_grid, source_returns, band_idx,
                          coherence_threshold=0.6, min_phase_strength=0.3,
                          smoothing=5):
    """
    Directional signals derived from the wavelet phase angle (lead/lag).

    This is the financially rigorous alternative to create_signals_from_resonance.
    Instead of blindly betting on reversion, it uses the phase angle to determine
    WHICH DIRECTION to trade and only acts when the signal asset demonstrably leads.

    Phase convention (pycwt.wct where first arg = traded asset, second = source):
      phase > 0  →  traded asset leads source  (source cannot predict traded asset)
      phase < 0  →  source leads traded asset  ← this is when we have an edge

    When source leads (phase clearly negative) and coherence is high:
      → Follow the source asset's recent trend direction for the traded asset.
      → If source trended up → long traded asset. If down → short it.

    Args:
        coherence_grid:   (n_freqs, n_times) from calculate_coherence
        phase_grid:       (n_freqs, n_times) phase angles in radians
        source_returns:   z-scored returns of the SIGNAL SOURCE (not traded) asset
        band_idx:         which frequency band to use
        coherence_threshold:  minimum coherence to generate a signal (default 0.6)
        min_phase_strength:   minimum |phase| in radians to confirm lead/lag (default 0.3 ≈ 17°)
        smoothing:        days to smooth phase and returns estimates (default 5)

    Returns: signal array (same length as coherence), values in {-1, 0, +1}
    """
    coherence = coherence_grid[band_idx, :]
    phase = phase_grid[band_idx, :]
    n = len(coherence)
    signals = np.zeros(n)

    for t in range(smoothing + 1, n):
        if coherence[t] < coherence_threshold:
            continue  # not coherent enough to trust the relationship

        # Smooth phase over recent window to reduce noise
        avg_phase = np.mean(phase[t - smoothing : t])

        # Only act when source is clearly leading (phase clearly negative)
        if avg_phase > -min_phase_strength:
            continue

        # Direction: follow recent trend of the leading source asset
        recent_source = np.sum(source_returns[max(0, t - smoothing) : t])
        signals[t] = 1.0 if recent_source > 0 else -1.0

    return signals


def apply_trend_filter(signals, prices, ma_period=50):
    """
    Macro regime filter: prevents the strategy from fighting the trend.

    - Long signals (+1) are zeroed out when price is below its MA.
    - Short signals (-1) are zeroed out when price is above its MA.

    Uses O(n) cumsum for efficiency (no per-bar loop).
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
