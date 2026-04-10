import numpy as np

def analyze_stance(bands, names):
    """
    Looks at the most recent points of each band to determine 
    if the cycle is currently pushing up (Bullish) or pulling down (Bearish).
    """
    stance_data = []
    
    for band, name in zip(bands, names):
        # Look at the last 5 days to determine recent slope
        recent_window = band[-5:]
        slope = np.polyfit(np.arange(len(recent_window)), recent_window, 1)[0]
        
        if name == "Underlying Structural Trend":
            weight = 0.5
        elif "Quarterly" in name:
            weight = 0.3
        elif "Monthly" in name:
            weight = 0.15
        else:
            weight = 0.05 # high frequency has low weight for 'stance'
            
        direction = "UP" if slope > 0 else "DOWN"
        strength = abs(slope)
        
        stance_data.append({
            "name": name,
            "direction": direction,
            "strength": strength,
            "weight": weight,
            "score": (1 if slope > 0 else -1) * weight
        })
        
    total_score = sum([s['score'] for s in stance_data])
    
    # Map score to english
    if total_score > 0.4:
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
