import numpy as np
from engine.backtest import run_backtest, create_signals_from_resonance

def test_run_backtest():
    # Setup some dummy data: mostly positive days, we stay long
    returns = np.array([0.01, 0.02, 0.01, -0.01, 0.03])
    actions = np.array([1, 1, 1, 1, 1])
    
    # Run the backtest simulation with zero slippage for easy checking
    results = run_backtest(returns, actions, slippage=0.0)
    
    # Our shift logic means the first day of returns is always 0 (can't trade the past)
    assert np.isclose(results['daily_returns'][0], 0.0)
    assert np.isclose(results['daily_returns'][1], 0.02)
    assert np.isclose(results['daily_returns'][2], 0.01)
    
    assert results['win_rate'] >= 0.0
    assert 'final_profit' in results

def test_create_signals_from_resonance():
    # Dummy grid: 2 frequency bands x 5 time steps
    resonance_grid = np.array([
        [0.8, 0.2, 0.5, 0.9, 0.1],
        [0.5, 0.5, 0.5, 0.5, 0.5]
    ])
    
    # Check the first band
    decisions = create_signals_from_resonance(resonance_grid, band_idx=0, high_barrier=0.7, low_barrier=0.3)
    
    # > 0.7 should signal -1, < 0.3 should signal 1
    expected_decisions = np.array([-1, 1, 0, -1, 1])
    np.testing.assert_array_equal(decisions, expected_decisions)
