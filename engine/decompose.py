import numpy as np
import pandas as pd
import pywt

def decompose_signal_mra(signal, wavelet='db4', level=5):
    """
    Perform Multi-Resolution Analysis (MRA) on a signal.
    Decomposes the signal into detailed levels and one approximation level.
    """
    # pywt.mra returns a list of arrays: [D_n, D_n-1, ..., D_1, A_n]
    # where D_i are details (high frequency) and A_n is approximation (low frequency trend)
    max_lvl = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_lvl)
    
    bands = pywt.mra(signal, wavelet, level=level)
    
    # We want to return them in a logical order: lowest frequency to highest frequency
    # A_n is the trend (lowest), D_n is next lowest, ..., D_1 is highest (noise)
    # The return list from mra is [D_L, D_L-1, ..., D_1, A_L]
    # Re-ordering to [A_L, D_L, D_L-1, ..., D_1]
    ordered_bands = [bands[-1]] + list(reversed(bands[:-1]))
    
    return ordered_bands

def get_band_labels(level):
    """
    Return labels for the frequency bands.
    """
    labels = ["Trend (Approx)"]
    for i in range(level, 0, -1):
        labels.append(f"Level {i} Detail")
    return labels

def consolidate_bands(bands, labels):
    """
    Create a DataFrame from bands and labels.
    """
    df = pd.DataFrame({label: band for label, band in zip(labels, bands)})
    return df

def verify_energy_conservation(original, reconstructed_bands):
    """
    Check if the sum of all bands equals the original signal.
    Wavelet decomposition is additive.
    """
    reconstructed = np.sum(reconstructed_bands, axis=0)
    return np.allclose(original, reconstructed)
