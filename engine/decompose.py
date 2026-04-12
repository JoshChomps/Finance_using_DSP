import pywt
import numpy as np
import pandas as pd

def slice_signal(signal, wavelet='db4', depth=5):
    """
    Decomposes a signal into distinct frequency bands via Multiresolution Analysis (MRA).

    Returns (bands, actual_depth).
    Bands are ordered from structural macro trend to high-frequency noise:
      [Approximation, Detail_depth, Detail_(depth-1), ..., Detail_1]
    """
    top_depth = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    actual_depth = min(depth, top_depth)

    clean_series = np.nan_to_num(np.squeeze(signal))
    original_len = len(clean_series)

    # Optimal padding to ensure transform alignment with 2^depth powers.
    padding_needed = (2**actual_depth) - (original_len % (2**actual_depth))
    if padding_needed == 2**actual_depth:
        padding_needed = 0

    padded_series = np.pad(clean_series, (0, padding_needed), 'reflect')

    # pywt.mra returns [approximation, detail_J, detail_(J-1), ..., detail_1]
    # No reordering required for coarsest-to-finest output.
    raw_bands = pywt.mra(padded_series, wavelet, level=actual_depth)
    trimmed   = [b[:original_len] for b in raw_bands]

    return trimmed, actual_depth


def create_labels(depth, sample_rate=1):
    """
    Generates technical identifiers for each frequency band based on spectral density.
    Mapped from structural macro trend (Approximation) to high-frequency volatility (Detail).

    periodicity_lo = 2^j / sample_rate
    periodicity_hi = 2^(j+1) / sample_rate
    """
    approx_cutoff = 2**depth // sample_rate
    labels = [f"Structural Trend (>{approx_cutoff}d)"]

    for level in range(depth, 0, -1):
        lo = 2**level // sample_rate
        hi = 2**(level + 1) // sample_rate

        if hi <= 4:
            tag = f"Micro-Volatility ({lo}-{hi}d)"
        elif hi <= 8:
            tag = f"Weekly Momentum ({lo}-{hi}d)"
        elif hi <= 16:
            tag = f"Bi-Weekly Cycle ({lo}-{hi}d)"
        elif hi <= 32:
            tag = f"Monthly Rhythm ({lo}-{hi}d)"
        elif hi <= 64:
            tag = f"Quarterly Swing ({lo}-{hi}d)"
        else:
            tag = f"Macro Cycle ({lo}-{hi}d)"

        labels.append(tag)

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
