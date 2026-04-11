import pytest
import numpy as np
import pandas as pd
from engine.decompose import slice_signal, create_labels
from engine.scalogram import run_cwt_analysis
from engine.coherence import calculate_coherence, compute_lead_lag_summary
from engine.granger import analyze_causal_flow

def test_decomposition():
    depth = 3
    length = 125
    signal = np.sin(np.linspace(0, 10, length)) + np.random.normal(0, 0.1, length)
    bands = slice_signal(signal, depth=depth)
    assert len(bands) == depth + 1
    assert all(len(b) == length for b in bands)

def test_create_labels_count():
    for depth in [2, 3, 5, 8]:
        labels = create_labels(depth)
        assert len(labels) == depth + 1, f"Expected {depth+1} labels for depth={depth}"

def test_create_labels_period_strings():
    labels = create_labels(5)
    # First label should mention the trend or macro concept
    assert "Macro" in labels[0] or "Trend" in labels[0] or "Structural" in labels[0]
    # Each non-trend label should reference a recognisable time period
    time_keywords = ("day", "days", "week", "weeks", "month", "months",
                     "noise", "swing", "micro", "cycle", "trend", "momentum")
    for lbl in labels[1:]:
        has_ref = any(kw in lbl.lower() for kw in time_keywords)
        assert has_ref, f"Label '{lbl}' should reference a time period"

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

def test_lead_lag_summary_types():
    """compute_lead_lag_summary should return a list of dicts with expected keys."""
    np.random.seed(1)
    t = np.linspace(0, 40, 512)   # longer signal avoids pycwt AR(1) edge case
    y1 = np.sin(t) + np.random.normal(0, 0.1, 512)
    y2 = np.sin(t + 0.3) + np.random.normal(0, 0.1, 512)
    cmap, phase, coi, freqs, _ = calculate_coherence(y1, y2)
    summary = compute_lead_lag_summary(phase, freqs, cmap, coi, min_coherence=0.3)
    assert isinstance(summary, list)
    for row in summary:
        assert "period_days" in row
        assert "lead_days"   in row
        assert "first_leads" in row
        assert isinstance(row["first_leads"], bool)

def test_granger():
    t = np.linspace(0, 50, 500)
    y = np.sin(t)
    x = np.sin(t - 1) + np.random.normal(0, 0.1, 500)
    data = np.vstack([x, y]).T
    freq_bins, flow_yx, flow_xy = analyze_causal_flow(data, maxlag=5)
    assert len(freq_bins) == len(flow_yx) == len(flow_xy)
    assert np.mean(flow_yx) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
