import numpy as np
import pytest
from engine.backtest import (
    run_backtest,
    create_signals_from_resonance,
    create_phase_signals,
    compute_kelly_fraction,
    apply_trend_filter,
    coherence_stability,
)


def test_run_backtest_basic():
    returns = np.array([0.01, 0.02, 0.01, -0.01, 0.03])
    actions = np.array([1, 1, 1, 1, 1])
    results = run_backtest(returns, actions, slippage=0.0)

    # 1-day execution lag: day 0 should always be zero
    assert np.isclose(results["daily_returns"][0], 0.0)
    assert np.isclose(results["daily_returns"][1], 0.02)
    assert np.isclose(results["daily_returns"][2], 0.01)
    assert results["win_rate"] >= 0.0
    assert "final_profit" in results
    assert "sharpe" in results


def test_run_backtest_position_size():
    returns = np.array([0.01, 0.02, 0.01, -0.01, 0.03])
    actions = np.array([1, 1, 1, 1, 1])
    full = run_backtest(returns, actions, slippage=0.0, position_size=1.0)
    half = run_backtest(returns, actions, slippage=0.0, position_size=0.5)
    assert np.isclose(half["daily_returns"][1], full["daily_returns"][1] * 0.5)
    assert np.isclose(half["daily_returns"][2], full["daily_returns"][2] * 0.5)


def test_create_signals_from_resonance():
    grid = np.array([
        [0.8, 0.2, 0.5, 0.9, 0.1],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    decisions = create_signals_from_resonance(grid, band_idx=0,
                                               high_barrier=0.7, low_barrier=0.3)
    expected = np.array([-1, 1, 0, -1, 1])
    np.testing.assert_array_equal(decisions, expected)


def test_compute_kelly_fraction_positive_edge():
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 500)
    signals = np.ones(500)
    kelly = compute_kelly_fraction(returns, signals, half=True)
    assert 0.0 < kelly <= 1.0, f"Expected kelly in (0, 1], got {kelly}"


def test_compute_kelly_fraction_no_edge():
    """Negative or zero edge must return 0.0, not 1.0."""
    returns = np.random.normal(-0.002, 0.02, 500)  # losing strategy
    signals = np.ones(500)
    kelly = compute_kelly_fraction(returns, signals, half=True)
    assert kelly == 0.0, f"Expected 0.0 for negative-edge strategy, got {kelly}"


def test_compute_kelly_fraction_insufficient_data():
    """Too few active bars -> return 0.0 (don't trade)."""
    returns = np.zeros(10)
    signals = np.ones(10)
    kelly = compute_kelly_fraction(returns, signals)
    assert kelly == 0.0


def test_apply_trend_filter():
    prices  = np.linspace(100, 200, 100)          # monotone uptrend
    signals = np.array([1] * 50 + [-1] * 50, dtype=float)
    filtered = apply_trend_filter(signals, prices, ma_period=20)

    longs  = np.sum(filtered == 1)
    shorts = np.sum(filtered == -1)
    assert longs > 0,   "Longs should survive in an uptrend"
    assert shorts < 50, "Shorts should be partially or fully blocked in an uptrend"


def test_coherence_stability_shape():
    np.random.seed(0)
    grid = np.random.rand(10, 200)  # 10 freq bands, 200 time steps
    stab = coherence_stability(grid, band_idx=0, window=50)
    assert len(stab) == 200
    # First `window` values should be NaN
    assert np.all(np.isnan(stab[:50]))
    # After warm-up, values should be non-negative
    assert np.all(stab[50:] >= 0)


def test_create_phase_signals_no_signal_when_phase_positive():
    """When phase is positive (traded asset leads), no signals should be generated."""
    n = 100
    coherence_grid = np.full((5, n), 0.8)   # high coherence always
    phase_grid     = np.full((5, n), 0.5)   # positive phase: traded leads source -> no edge
    source_returns = np.ones(n) * 0.01

    signals = create_phase_signals(
        coherence_grid, phase_grid, source_returns,
        band_idx=2, coherence_threshold=0.6, min_phase_strength=0.3,
    )
    assert np.sum(signals != 0) == 0, "Should generate no signals when source is the follower"


def test_create_phase_signals_follows_source_direction():
    """When source leads (negative phase) and coherence is high, follow source direction."""
    n = 100
    coherence_grid = np.full((5, n), 0.8)
    phase_grid     = np.full((5, n), -0.6)   # clearly negative: source leads
    # Source has been going up every day
    source_returns = np.full(n, 0.005)

    signals = create_phase_signals(
        coherence_grid, phase_grid, source_returns,
        band_idx=2, coherence_threshold=0.6, min_phase_strength=0.3, smoothing=5,
    )
    active = signals[signals != 0]
    assert len(active) > 0, "Should generate signals when source leads"
    assert np.all(active == 1.0), "Signals should be long (+1) when source is trending up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
