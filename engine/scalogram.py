import numpy as np
import pandas as pd
from ssqueezepy import cwt, ssq_cwt, Wavelet
from scipy.signal import stft, welch

def run_cwt_analysis(signal, wavelet='morlet'):
    """
    Continuous Wavelet Transform (CWT).
    Helps us see how the frequency content of a stock changes over time.
    """
    map_complex, scales = cwt(signal, wavelet)
    return map_complex, scales

def run_synchrosqueezing(signal, wavelet='morlet'):
    """
    A more advanced version of wavelet analysis that "tightens up" 
    the energy to give us sharper frequency resolution.
    """
    tight_map, raw_map, ssq_freqs, scales = ssq_cwt(signal, wavelet)
    return tight_map, raw_map, ssq_freqs, scales

def track_frequency_flow(signal, sample_rate=1.0, window_size=64):
    """
    Short-Time Fourier Transform (STFT). 
    Useful for seeing quick shifts in volatility and noise patterns.
    """
    freqs, times, map_z = stft(signal, fs=sample_rate, nperseg=window_size)
    return freqs, times, map_z

def estimate_power_spectrum(signal, sample_rate=1.0):
    """
    Estimates which frequencies hold the most energy in the signal.
    """
    freqs, energy_density = welch(signal, fs=sample_rate)
    return freqs, energy_density

def get_magnitude(complex_data):
    """
    Just a helper to get the absolute value (intensity) of complex numbers.
    """
    return np.abs(complex_data)

