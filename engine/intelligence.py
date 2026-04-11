import numpy as np

def analyze_stance(bands, names):
    """
    Looks at the most recent points of each band to determine 
    if the cycle is currently pushing up (Bullish) or pulling down (Bearish).
    """
    stance_data = []
    
    for band, name in zip(bands, names):
        # Look at the last 10 days to determine recent slope for better stability
        lookback = 10
        recent_window = band[-lookback:]
        x = np.arange(len(recent_window))
        slope, intercept = np.polyfit(x, recent_window, 1)
        
        # Normalize slope by band volatility to get 'z-slope' (strength relative to band variance)
        band_std = np.std(band) if np.std(band) > 0 else 1e-6
        normalized_strength = slope / band_std
        
        if name == "Underlying Structural Trend":
            weight = 0.50
        elif "Quarterly" in name:
            weight = 0.25
        elif "Monthly" in name:
            weight = 0.15
        elif "Weekly" in name:
            weight = 0.07
        else:
            weight = 0.03 # High frequency has minimal impact on 'stance'
            
        direction = "UP" if slope > 0 else "DOWN"
        
        # Score is the weighted normalized strength
        stance_score = np.clip(normalized_strength * weight * 2, -weight, weight)
        
        stance_data.append({
            "name": name,
            "direction": direction,
            "strength": abs(normalized_strength),
            "weight": weight,
            "score": stance_score
        })
        
    total_score = sum([s['score'] for s in stance_data])
    
    # Map score to english
    if total_score > 0.25:
        label = "Strong Bullish"
    elif total_score > 0.1:
        label = "Bullish / Accumulating"
    elif total_score < -0.4:
        label = "Strong Bearish"
    elif total_score < -0.1:
        label = "Bearish / Distribution"
    else:
        label = "Neutral / Consolidating"
        
    return label, total_score, stance_data

def project_structural_trend(trend_band, horizon=10):
    """
    Extrapolates the structural trend band into the future.
    """
    recent_len = 20
    series = trend_band[-recent_len:]
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series, 1) # Simple linear fit of the structural trend
    
    future_x = np.arange(len(series), len(series) + horizon)
    projection = np.polyval(coeffs, future_x)
    
    return projection
