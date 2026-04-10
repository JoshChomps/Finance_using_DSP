import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests

def check_standard_causality(data, lags=5):
    """
    Checks if one series predicts the future of another using 
    standard time-domain regression.
    """
    test_results = grangercausalitytests(data, maxlag=lags, verbose=False)
    # just pull out the p-values for the main test
    results_map = {lag: test_results[lag][0]['ssr_ftest'][1] for lag in test_results}
    return results_map

def analyze_causal_flow(data, maxlag=5, resolution=100):
    """
    Measures the strength of leadership between two assets across 
    different cycles (frequencies).
    """
    rows, cols = data.shape
    if cols != 2:
        raise ValueError("We only support comparing two variables right now.")

    # 1. Fit the Vector Auto-Regression model
    var_model = VAR(data)
    fit_result = var_model.fit(maxlag)
    lag_order = fit_result.k_ar
    
    # Grab the model coefficients and the errors
    coefficients = fit_result.coefs
    resid_cov = fit_result.sigma_u
    
    # 2. Setup the frequency range (normalized 0 to 0.5)
    freq_bins = np.linspace(0, 0.5, resolution)
    
    flow_yx = np.zeros(resolution)
    flow_xy = np.zeros(resolution)
    
    for i, f in enumerate(freq_bins):
        # 3. Calculate the Transfer Function H at this frequency
        identity = np.eye(cols, dtype=complex)
        for k in range(1, lag_order + 1):
            identity -= coefficients[k-1] * np.exp(-1j * 2 * np.pi * f * k)
        
        transfer_matrix = np.linalg.inv(identity)
        
        # 4. Get the Spectral Density Matrix
        spectrum = transfer_matrix @ resid_cov @ transfer_matrix.conj().T
        
        # 5. Geweke's measure of causality
        var_x = resid_cov[0, 0]
        var_y = resid_cov[1, 1]
        cov_xy = resid_cov[0, 1]
        
        h_xy = transfer_matrix[0, 1]
        h_yx = transfer_matrix[1, 0]
        
        spec_xx = spectrum[0, 0].real
        spec_yy = spectrum[1, 1].real
        
        # Avoid division by zero or negative logs with a small epsilon
        denom_yx = spec_xx - (var_y - cov_xy**2/var_x) * np.abs(h_xy)**2
        denom_xy = spec_yy - (var_x - cov_xy**2/var_y) * np.abs(h_yx)**2
        
        flow_yx[i] = np.log(spec_xx / max(1e-12, denom_yx))
        flow_xy[i] = np.log(spec_yy / max(1e-12, denom_xy))

    return freq_bins, flow_yx, flow_xy

def interpret_causality(p_values, limit=0.05):
    """
    Translates raw p-values into a simple sentence.
    """
    leads = [lag for lag, p in p_values.items() if p < limit]
    if leads:
        return f"Statistically significant leadership found at lags: {leads}"
    return "No clear leadership detected in this window."
