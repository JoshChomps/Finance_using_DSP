import numpy as np
import pandas as pd
from ssqueezepy import cwt, ssq_cwt, Wavelet
from scipy.signal import stft, welch

def compute_cwt(signal, wavelet='morlet'):
    """
    Compute Continuous Wavelet Transform (CWT).
    Returns (Wx, scales)
    """
    # ssqueezepy cwt expects a wavelet object or string
    Wx, scales = cwt(signal, wavelet)
    return Wx, scales

def compute_ssq_cwt(signal, wavelet='morlet'):
    """
    Compute Synchrosqueezed Continuous Wavelet Transform (SSQ-CWT).
    Returns (Tx, Wx, ssq_freqs, scales)
    """
    Tx, Wx, ssq_freqs, scales = ssq_cwt(signal, wavelet)
    return Tx, Wx, ssq_freqs, scales

def compute_stft(signal, fs=1.0, nperseg=64):
    """
    Compute Short-Time Fourier Transform (STFT).
    Returns (f, t, Zxx)
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    return f, t, Zxx

def compute_psd(signal, fs=1.0):
    """
    Compute Power Spectral Density (PSD) using Welch's method.
    Returns (f, Pxx)
    """
    f, Pxx = welch(signal, fs=fs)
    return f, Pxx

def get_magnitude(complex_coeffs):
    """
    Return the magnitude of complex transform coefficients.
    """
    return np.abs(complex_coeffs)
