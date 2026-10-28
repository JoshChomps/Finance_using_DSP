import numpy as np
import pandas as pd
from ssqueezepy import cwt, ssq_cwt, Wavelet
from scipy.signal import stft, welch

def run_cwt_analysis(signal, wavelet='morlet'):
    """
    Continuous Wavelet Transform (CWT).
    Localizes time-frequency energy distribution for non-stationary market signals.
    """
    eps = 1e-12
    map_complex, scales = cwt(signal, wavelet)
    return map_complex + eps, scales

def run_synchrosqueezing(signal, wavelet='morlet'):
    """
    Synchrosqueezed Wavelet Transform (SWT).
    Reassigns energy to frequency points for high-resolution spectral identification.
    """
    tight_map, raw_map, ssq_freqs, scales = ssq_cwt(signal, wavelet)
    return tight_map, raw_map, ssq_freqs, scales

def track_frequency_flow(signal, sample_rate=1.0, window_size=64):
    """
    Short-Time Fourier Transform (STFT). 
    Quantifies frequency-domain leadership and noise-spectrum shifts over discrete segments.
    """
    freqs, times, map_z = stft(signal, fs=sample_rate, nperseg=window_size)
    return freqs, times, map_z

def estimate_power_spectrum(signal, sample_rate=1.0):
    """
    Welch Power Spectral Density (PSD) estimation.
    Calculates the average distribution of signal variance across the frequency spectrum.
    """
    freqs, energy_density = welch(signal, fs=sample_rate)
    return freqs, energy_density

def get_magnitude(complex_data):
    """Institutional magnitude extraction with epsilon safety Floor."""
    return np.abs(complex_data) + 1e-12

