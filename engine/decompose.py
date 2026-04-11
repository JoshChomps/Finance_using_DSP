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

    # pywt.mra returns [approx, detail_J, detail_(J-1), ..., detail_1]
    # i.e. coarsest-to-finest (approximation first). No reordering needed.
    raw_bands = pywt.mra(padded_series, wavelet, level=depth)
    trimmed   = [b[:original_len] for b in raw_bands]

    return trimmed


def create_labels(depth, sample_rate=1):
    """
    Generates human-readable names for each frequency band, including the
    approximate period range in trading days (assumes daily data by default).

    sample_rate: samples per day (1 for daily, 390 for 1-minute, etc.)

    MODWT detail at level j captures roughly [2^j, 2^(j+1)] samples per cycle.
    Dividing by sample_rate converts to calendar days.

    Example for depth=5, daily data:
      Macro Trend      → >32 days
      Quarterly Cycle  → 32–64 days   (detail 5)
      Monthly Cycle    → 16–32 days   (detail 4)
      Bi-Weekly Cycle  → 8–16 days    (detail 3)
      Weekly Cycle     → 4–8 days     (detail 2)
      Noise            → 2–4 days     (detail 1)
    """
    approx_cutoff = 2**depth // sample_rate
    names = [f"Macro Trend (>{approx_cutoff}d)"]

    for level in range(depth, 0, -1):
        lo = 2**level // sample_rate
        hi = 2**(level + 1) // sample_rate

        if hi <= 4:
            tag = f"Noise ({lo}–{hi}d)"
        elif hi <= 8:
            tag = f"Weekly Cycle ({lo}–{hi}d)"
        elif hi <= 16:
            tag = f"Bi-Weekly Cycle ({lo}–{hi}d)"
        elif hi <= 32:
            tag = f"Monthly Cycle ({lo}–{hi}d)"
        elif hi <= 64:
            tag = f"Quarterly Cycle ({lo}–{hi}d)"
        else:
            tag = f"Long Cycle ({lo}–{hi}d)"

        names.append(tag)

    return names


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
