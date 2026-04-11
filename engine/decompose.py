import numpy as np
import pandas as pd
import pywt


def slice_signal(signal, wavelet='db4', depth=5):
    """
    Splits a signal into different frequency levels via Multiresolution Analysis (MRA).
    Think of it like separating a song into bass, mids, and treble.

    Returns bands ordered from slowest (trend) to fastest (noise):
      [Approximation, Detail_depth, Detail_(depth-1), ..., Detail_1]
    """
    top_depth = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    depth = min(depth, top_depth)

    clean_series = np.nan_to_num(np.squeeze(signal))
    original_len = len(clean_series)

    # Pad to a multiple of 2^depth so the wavelet math works cleanly,
    # then trim the padding back after the transform.
    padding_needed = (2**depth) - (original_len % (2**depth))
    if padding_needed == 2**depth:
        padding_needed = 0

    padded_series = np.pad(clean_series, (0, padding_needed), 'reflect')

    # pywt.mra returns [detail_1, detail_2, ..., detail_depth, approx]
    # i.e. finest-to-coarsest; we reorder to coarsest-to-finest.
    raw_bands = pywt.mra(padded_series, wavelet, level=depth)
    trimmed   = [b[:original_len] for b in raw_bands]

    # [approx, detail_depth, detail_(depth-1), ..., detail_1]
    return [trimmed[-1]] + list(reversed(trimmed[:-1]))


def create_labels(depth, sample_rate=1):
    """
    Generates human-readable names for each frequency band, ordered from
    slowest (Approximation / Structural Trend) to fastest (Noise).

    Uses intuitive market-cycle names for the common depth=5 daily case,
    and falls back to period-range notation for non-standard depths.

    MODWT detail at level j captures roughly [2^j, 2^(j+1)] samples per cycle.
    Dividing by sample_rate converts to calendar days.
    """
    labels = ["Underlying Structural Trend"]

    # Detail bands go from depth (slowest) down to 1 (fastest)
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
            labels.append("Micro-Noise (1-2 Days)")
        elif hi <= 64:
            labels.append(f"Macro Cycle ({lo}–{hi}d)")
        else:
            labels.append(f"Deep Macro Cycle ({lo}–{hi}d)")

    return labels


def map_to_dataframe(bands, names):
    """Wraps raw band arrays into a labelled DataFrame."""
    return pd.DataFrame({name: b for name, b in zip(names, bands)})


def check_reconstruction(original, components):
    """
    Sanity check: do the pieces still add up to the whole?
    Validates that the MRA decomposition is perfectly invertible.
    """
    reconstructed = np.sum(components, axis=0)
    return np.allclose(original, reconstructed)
