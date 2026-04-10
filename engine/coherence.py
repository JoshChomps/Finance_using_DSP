import numpy as np
import pandas as pd
import pycwt as wavelet
from pycwt.helpers import find

def calculate_coherence(first_series, second_series, time_step=1.0, scale_resolution=1/12):
    """
    Measures how much two signals "resonate" with each other at different 
    frequencies and points in time.
    """
    # use the Morlet wavelet as the "mother" wave
    standard_wave = wavelet.Morlet()
    
    # Calculate Cross Wavelet Transform (measuring shared power)
    cross_power, cone_of_influence, frequencies, significance = wavelet.xwt(
        first_series, second_series, time_step, dj=scale_resolution, wavelet=standard_wave
    )
    
    # Calculate the actual coherence map and the phase relationship (leading/lagging)
    coherence_map, phase_angle, coi, freqs, sig = wavelet.wct(
        first_series, second_series, time_step, dj=scale_resolution, wavelet=standard_wave, sig=False
    )
    
    return coherence_map, phase_angle, coi, freqs, sig

def check_coherence_significance(a, b, dt=1.0, dj=1/12):
    """
    Checks if the observed resonance is statistically significant 
    or just random noise.
    """
    wave = wavelet.Morlet()
    return wavelet.wct_significance(a, b, dt, dj=dj, mother=wave)

