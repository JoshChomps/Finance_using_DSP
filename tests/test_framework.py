import pytest
import numpy as np
import pandas as pd
from engine.decompose import slice_signal
from engine.scalogram import run_cwt_analysis
from engine.coherence import calculate_coherence
from engine.granger import analyze_causal_flow

def test_decomposition():
    # Signal length logic: our slice_signal pads automatically now
    depth = 3
    length = 125 
    signal = np.sin(np.linspace(0, 10, length)) + np.random.normal(0, 0.1, length)
    bands = slice_signal(signal, depth=depth)
    assert len(bands) == depth + 1
    assert all(len(b) == length for b in bands)

def test_scalogram():
    signal = np.sin(np.linspace(0, 10, 128)) + np.random.normal(0, 0.1, 128)
    map_complex, scales = run_cwt_analysis(signal)
    assert map_complex.shape[1] == 128
    assert len(scales) == map_complex.shape[0]

def test_coherence():
    np.random.seed(42)
    t = np.linspace(0, 10, 256)
    y1 = np.sin(t) + np.random.normal(0, 0.2, 256)
    y2 = np.sin(t + 0.5) + np.random.normal(0, 0.2, 256)
    resonance_map, phase, coi, freqs, sig = calculate_coherence(y1, y2)
    assert resonance_map.shape[1] == 256
    assert np.all(resonance_map >= 0)

def test_granger():
    # Simple leader-follower relation simulation
    t = np.linspace(0, 50, 500)
    y = np.sin(t)
    x = np.sin(t - 1) + np.random.normal(0, 0.1, 500) # y leads x
    data = np.vstack([x, y]).T
    freq_bins, flow_yx, flow_xy = analyze_causal_flow(data, maxlag=5)
    
    # Success if it returns arrays of correct size
    assert len(freq_bins) == len(flow_yx) == len(flow_xy)
    assert np.mean(flow_yx) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
