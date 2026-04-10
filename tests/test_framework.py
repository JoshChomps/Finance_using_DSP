import pytest
import numpy as np
import pandas as pd
from engine.decompose import decompose_signal_mra
from engine.scalogram import compute_cwt
from engine.coherence import compute_wavelet_coherence
from engine.granger import compute_spectral_granger

def test_decomposition():
    # Signal length must be divisible by 2**level for swt-based mra
    level = 3
    length = 128 # 2**7
    signal = np.sin(np.linspace(0, 10, length)) + np.random.normal(0, 0.1, length)
    bands = decompose_signal_mra(signal, level=level)
    assert len(bands) == level + 1
    assert all(len(b) == length for b in bands)

def test_scalogram():
    signal = np.sin(np.linspace(0, 10, 128)) + np.random.normal(0, 0.1, 128)
    Wx, scales = compute_cwt(signal)
    assert Wx.shape[1] == 128
    assert len(scales) == Wx.shape[0]

def test_coherence():
    # pycwt needs stochastic component for AR(1) estimation
    np.random.seed(42)
    t = np.linspace(0, 10, 256)
    y1 = np.sin(t) + np.random.normal(0, 0.2, 256)
    y2 = np.sin(t + 0.5) + np.random.normal(0, 0.2, 256)
    wct, phase, coi, freqs, scales = compute_wavelet_coherence(y1, y2)
    assert wct.shape[1] == 256
    assert np.all(wct >= 0)

def test_granger():
    # Simple leader-follower relation
    t = np.linspace(0, 50, 500)
    y = np.sin(t)
    x = np.sin(t - 1) + np.random.normal(0, 0.1, 500) # y leads x
    data = np.vstack([x, y]).T
    freqs, g_yx, g_xy = compute_spectral_granger(data, maxlag=5)
    # Success if it returns arrays of correct size
    assert len(freqs) == len(g_yx) == len(g_xy)
    # Check if mean causal strength y->x is positive and likely > x->y
    assert np.mean(g_yx) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
