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
    Measures the directional leadership (Geweke Causality) between two assets 
    across the frequency domain (0 to 0.5 Hz).
    
    Decomposes total spectral power into:
    Total Power = Intrinsic (Unpredictable) Power + Causal (Predictable) Power.
    """
    rows, cols = data.shape
    if cols != 2:
        raise ValueError("Dimensional mismatch: analyze_causal_flow requires a bivariate system.")

    # 1. Fit Vector Auto-Regression (VAR) Model
    var_model = VAR(data)
    fit_result = var_model.fit(maxlag)
    lag_order = fit_result.k_ar
    coefficients = fit_result.coefs
    resid_cov = fit_result.sigma_u
    
    # 2. Setup Frequency Grid
    freq_bins = np.linspace(0, 0.5, resolution)
    flow_yx = np.zeros(resolution)
    flow_xy = np.zeros(resolution)
    
    # 3. Frequency-Domain Decomposition
    for i, f in enumerate(freq_bins):
        # Calculate Transfer Matrix H(f) = [I - sum(A_k * e^-i2pikf)]^-1
        identity = np.eye(cols, dtype=complex)
        for k in range(1, lag_order + 1):
            identity -= coefficients[k-1] * np.exp(-1j * 2 * np.pi * f * k)
        
        try:
            transfer_matrix = np.linalg.inv(identity)
        except np.linalg.LinAlgError:
            continue # Skip singular frequencies
        
        # Spectral Density Matrix S(f) = H(f) * Sigma * H(f)*
        spectrum = transfer_matrix @ resid_cov @ transfer_matrix.conj().T
        
        # 4. Geweke's Measure: Contrast Total Power with Intrinsic Power
        # Intrinsic Power is the portion of the spectrum NOT attributable to the co-asset.
        var_x = resid_cov[0, 0]
        var_y = resid_cov[1, 1]
        cov_xy = resid_cov[0, 1]
        
        h_xy = transfer_matrix[0, 1]
        h_yx = transfer_matrix[1, 0]
        
        spec_xx = max(1e-12, spectrum[0, 0].real)
        spec_yy = max(1e-12, spectrum[1, 1].real)

        # Causal Contribution from Y to X
        # denom = Intrinsic Power of X
        denom_yx = spec_xx - (var_y - cov_xy**2/max(1e-12, var_x)) * np.abs(h_xy)**2
        # Causal Contribution from X to Y
        denom_xy = spec_yy - (var_x - cov_xy**2/max(1e-12, var_y)) * np.abs(h_yx)**2

        flow_yx[i] = max(0.0, np.log(spec_xx / max(1e-12, denom_yx)))
        flow_xy[i] = max(0.0, np.log(spec_yy / max(1e-12, denom_xy)))

    return freq_bins, flow_yx, flow_xy

def interpret_causality(p_values, limit=0.05):
    """
    Translates raw p-values into a simple sentence.
    """
    leads = [lag for lag, p in p_values.items() if p < limit]
    if leads:
        return f"Statistically significant leadership found at lags: {leads}"
    return "No clear leadership detected in this window."
