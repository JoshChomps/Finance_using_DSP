import numpy as np
from engine.backtest import (
    run_backtest,
    create_signals_from_resonance,
    compute_kelly_fraction,
    apply_trend_filter,
)


def test_run_backtest():
    # Setup some dummy data: mostly positive days, we stay long
    returns = np.array([0.01, 0.02, 0.01, -0.01, 0.03])
    actions = np.array([1, 1, 1, 1, 1])

    # Run the backtest simulation with zero slippage for easy checking
    results = run_backtest(returns, actions, slippage=0.0)

    # Our shift logic means the first day of returns is always 0
    assert np.isclose(results['daily_returns'][0], 0.0)
    assert np.isclose(results['daily_returns'][1], 0.02)
    assert np.isclose(results['daily_returns'][2], 0.01)

    assert results['win_rate'] >= 0.0
    assert 'final_profit' in results
    assert 'position_size' in results


def test_run_backtest_position_size():
    returns = np.array([0.01, 0.02, 0.01, -0.01, 0.03])
    actions = np.array([1, 1, 1, 1, 1])

    full = run_backtest(returns, actions, slippage=0.0, position_size=1.0)
    half = run_backtest(returns, actions, slippage=0.0, position_size=0.5)

    # Half-sized position should produce roughly half the daily P&L on active days
    assert np.isclose(half['daily_returns'][1], full['daily_returns'][1] * 0.5)
    assert np.isclose(half['daily_returns'][2], full['daily_returns'][2] * 0.5)


def test_create_signals_from_resonance():
    # Dummy grid: 2 frequency bands x 5 time steps
    resonance_grid = np.array([
        [0.8, 0.2, 0.5, 0.9, 0.1],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ])

    decisions = create_signals_from_resonance(resonance_grid, band_idx=0,
                                               high_barrier=0.7, low_barrier=0.3)

    # > 0.7 → -1 (short),  < 0.3 → +1 (long),  otherwise 0
    expected = np.array([-1, 1, 0, -1, 1])
    np.testing.assert_array_equal(decisions, expected)


def test_compute_kelly_fraction():
    np.random.seed(42)
    # Positive-expectancy strategy: slight drift upward
    returns = np.random.normal(0.001, 0.02, 500)
    signals = np.ones(500)

    kelly = compute_kelly_fraction(returns, signals, half=True)
    assert 0.0 < kelly <= 1.0, f"Kelly should be in (0, 1], got {kelly}"

    # Zero-edge strategy should return a clamped value (not negative/zero)
    flat_returns = np.zeros(500)
    kelly_flat = compute_kelly_fraction(flat_returns, signals, half=True)
    assert kelly_flat == 1.0  # fallback when no positive edge


def test_apply_trend_filter():
    # Prices trend upward the whole way
    prices = np.linspace(100, 200, 100)
    # 50 longs and 50 shorts
    signals = np.array([1] * 50 + [-1] * 50, dtype=float)

    filtered = apply_trend_filter(signals, prices, ma_period=20)

    # In a strong uptrend the price stays above the MA almost immediately,
    # so short signals should be removed and longs kept
    longs  = np.sum(filtered == 1)
    shorts = np.sum(filtered == -1)
    assert longs > 0,  "Expected some long signals to survive"
    assert shorts < 50, "Expected trend filter to block some/all short signals in uptrend"
