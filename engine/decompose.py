import numpy as np
import pandas as pd
import pywt


def slice_signal(signal, wavelet='db4', depth=5):
    """
    Decomposes a signal into distinct frequency bands via Multiresolution Analysis (MRA).

    Returns (bands, actual_depth).
    Bands are ordered from structural macro trend to high-frequency noise:
      [Approximation, Detail_depth, Detail_(depth-1), ..., Detail_1]
    """
    top_depth = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    depth = min(depth, top_depth)

    clean_series = np.nan_to_num(np.squeeze(signal))
    original_len = len(clean_series)

    # Signal padding to ensure transform alignment with 2^depth.
    padding_needed = (2**depth) - (original_len % (2**depth))
    if padding_needed == 2**depth:
        padding_needed = 0

    padded_series = np.pad(clean_series, (0, padding_needed), 'reflect')

    # pywt.mra returns [approximation, detail_J, detail_(J-1), ..., detail_1]
    bands = pywt.mra(padded_series, wavelet, level=depth)
    trimmed = [b[:original_len] for b in bands]

    return trimmed, depth


def create_labels(depth, sample_rate=1):
    """
    Generates technical identifiers for each frequency band.
    Ordered from structural trend (Approximation) to high-frequency volatility (Detail).

    Detail components at level j correspond to [2^j, 2^(j+1)] units per cycle.
    """
    labels = ["Underlying Structural Trend"]

    for i in range(depth, 0, -1):
        lo = 2**i // sample_rate
        hi = 2**(i + 1) // sample_rate

        if i == 5:
            labels.append("Quarterly Trend (1-3 Months)")
        elif i == 4:
            labels.append("Monthly Momentum (2-4 Weeks)")
        elif i == 3:
            labels.append("Weekly Cycles (1-2 Weeks)")
        elif i == 2:
            labels.append("Fast Swings (2-4 Days)")
        elif i == 1:
            labels.append("Micro-Volatility (1-2 Days)")
        elif hi <= 64:
            labels.append(f"Macro Cycle ({lo} to {hi} days)")
        else:
            labels.append(f"Deep Macro Cycle ({lo} to {hi} days)")

    return labels


def map_to_dataframe(bands, names):
    """Maps spectral components to a labelled DataFrame."""
    return pd.DataFrame({name: b for name, b in zip(names, bands)})


def check_reconstruction(original, components):
    """
    Validates signal reconstruction integrity.
    Verifies that the MRA decomposition is mathematically invertible.
    """
    reconstructed = np.sum(components, axis=0)
    return np.allclose(original, reconstructed)
