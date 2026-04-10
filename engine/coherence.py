import numpy as np
import pandas as pd
import pycwt as wavelet
from pycwt.helpers import find

def compute_wavelet_coherence(y1, y2, dt=1.0, dj=1/12):
    """
    Compute wavelet coherence between two signals.
    Returns (WCT, phase, units, period, scale, coi)
    """
    # y1, y2 should be normalized signals
    # pycwt implementation
    
    # Cross Wavelet Transform
    # mother = Morlet() is standard for coherence
    mother = wavelet.Morlet()
    
    # Cross Wavelet Transform
    # pycwt.xwt(y1, y2, dt, dj, s0, J, wavelet)
    # returns xwt, coi, freqs, significance
    x_wt, coi, freqs, signi = wavelet.xwt(y1, y2, dt, dj=dj, wavelet=mother)
    
    # Wavelet Coherence Transform
    # wct(y1, y2, dt, dj, s0, J, sig, significance_level, wavelet, normalize)
    # returns WCT, aWCT, coi, freqs, sig
    wct, phase, coi, freqs, sig = wavelet.wct(y1, y2, dt, dj=dj, wavelet=mother, sig=False)
    
    return wct, phase, coi, freqs, sig

def compute_significance(y1, y2, dt=1.0, dj=1/12):
    """
    Compute wavelet coherence significance.
    Note: This is computationally expensive as it uses MC simulations.
    """
    mother = wavelet.Morlet()
    # Estimate the significance level
    # We use a lower number of iterations for hackathon speed if needed, 
    # but pycwt has good defaults.
    sig = wavelet.wct_significance(y1, y2, dt, dj=dj, mother=mother)
    return sig

def get_coherence_matrix(wct_coeffs):
    """
    Return the magnitude squared coherence (the actual WTC measure).
    """
    return wct_coeffs
