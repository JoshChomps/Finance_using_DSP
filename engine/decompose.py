import numpy as np
import pandas as pd
import pywt

def slice_signal(signal, wavelet='db4', depth=5):
    """
    Splits a signal into different frequency levels (MRA).
    Think of it like separating a song into bass, mids, and treble.
    """
    # Figure out the max depth possible for this data length
    top_depth = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    depth = min(depth, top_depth)
    
    # Handle any weird shapes or NaNs
    clean_series = np.nan_to_num(np.squeeze(signal))
    original_len = len(clean_series)
    
    # Wavelet math needs specific lengths (multiples of 2^depth)
    # We'll pad it, transform it, then trim it back
    padding_needed = (2**depth) - (original_len % (2**depth))
    if padding_needed == 2**depth:
        padding_needed = 0
        
    padded_series = np.pad(clean_series, (0, padding_needed), 'reflect')
    
    raw_bands = pywt.mra(padded_series, wavelet, level=depth)
    
    # Trim the padding away
    trimmed_bands = [b[:original_len] for b in raw_bands]
    
    # Order them from slow trends to fast noise.
    # pywt returns [Details... Approximation], we want [Approximation, Details...]
    final_bands = [trimmed_bands[-1]] + list(reversed(trimmed_bands[:-1]))
    
    return final_bands

def create_labels(depth):
    """
    Generates names for the bands so we know which is which.
    """
    names = ["Long-term Trend"]
    for i in range(depth, 0, -1):
        names.append(f"Cycle Level {i}")
    return names

def map_to_dataframe(bands, names):
    """
    Wraps the raw arrays into a nice DataFrame.
    """
    return pd.DataFrame({name: b for name, b in zip(names, bands)})

def check_reconstruction(original, components):
    """
    Sanity check: do the pieces still add up to the whole?
    """
    reconstructed = np.sum(components, axis=0)
    return np.allclose(original, reconstructed)
