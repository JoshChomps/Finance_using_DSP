import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests

def compute_time_domain_granger(data, maxlag=5):
    """
    Compute standard time-domain Granger causality.
    Returns a dictionary of p-values.
    """
    results = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    # Extract p-values for the F-test (usually index 0 in the test result tuple)
    p_values = {lag: results[lag][0]['ssr_ftest'][1] for lag in results}
    return p_values

def compute_spectral_granger(data, maxlag=5, n_freqs=100):
    """
    Compute frequency-domain Granger causality (Geweke 1982).
    Input: data (2D array [T, 2])
    Output: freqs, G_yx (y leads x), G_xy (x leads y)
    """
    T, N = data.shape
    if N != 2:
        raise ValueError("Spectral Granger currently only supports bivariate analysis (2 variables).")

    # 1. Fit VAR model
    model = VAR(data)
    results = model.fit(maxlag)
    p = results.k_ar
    
    # Coefficients and residual covariance
    # coefs: [p, N, N]
    coefs = results.coefs
    sigma = results.sigma_u # Already a numpy array
    
    # 2. Setup frequency range
    freqs = np.linspace(0, 0.5, n_freqs) # Normalized frequency [0, 0.5]
    
    g_yx = np.zeros(n_freqs)
    g_xy = np.zeros(n_freqs)
    
    for i, f in enumerate(freqs):
        # 3. Compute Transfer Function H(f)
        # H(f) = (I - sum(A_k * exp(-i*2*pi*f*k)))^-1
        A_f = np.eye(N, dtype=complex)
        for k in range(1, p + 1):
            A_f -= coefs[k-1] * np.exp(-1j * 2 * np.pi * f * k)
        
        H = np.linalg.inv(A_f)
        
        # 4. Compute Spectral Matrix S(f)
        S = H @ sigma @ H.conj().T
        
        # 5. Compute Geweke measure (bivariate formulation)
        # G_y->x(f)
        # Formula: ln( S_xx / (S_xx - (sig_yy - sig_xy^2/sig_xx) * |H_xy|^2) )
        sig_xx = sigma[0, 0]
        sig_yy = sigma[1, 1]
        sig_xy = sigma[0, 1]
        
        h_xy = H[0, 1]
        h_yx = H[1, 0]
        
        s_xx = S[0, 0].real
        s_yy = S[1, 1].real
        
        # Ensure values stay within valid ranges for log
        denom_yx = s_xx - (sig_yy - sig_xy**2/sig_xx) * np.abs(h_xy)**2
        denom_xy = s_yy - (sig_xx - sig_xy**2/sig_yy) * np.abs(h_yx)**2
        
        g_yx[i] = np.log(s_xx / max(1e-12, denom_yx))
        g_xy[i] = np.log(s_yy / max(1e-12, denom_xy))

    return freqs, g_yx, g_xy

def get_causality_interpretation(p_values, threshold=0.05):
    """
    Convert p-values to a human-readable interpretation.
    """
    significant_lags = [lag for lag, p in p_values.items() if p < threshold]
    if significant_lags:
        return f"Significant causality detected at lags: {significant_lags}"
    return "No significant time-domain causality detected."
