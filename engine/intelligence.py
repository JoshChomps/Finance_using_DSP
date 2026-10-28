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
    
    # Regime Detection Logic
    structural_bias = stance_data[0]['score'] / stance_data[0]['weight'] # Unweighted normalized strength
    detail_alignment = np.mean([s['score'] / s['weight'] for s in stance_data[1:min(4, len(stance_data))]])
    
    # 1. Primary Classification (The Stance)
    if total_score > 0.25:
        base_label = "Strong Bullish"
    elif total_score > 0.05:
        base_label = "Bullish / Accumulating"
    elif total_score < -0.3:
        base_label = "Strong Bearish"
    elif total_score < -0.05:
        base_label = "Bearish / Distribution"
    else:
        base_label = "Neutral / Consolidating"
        
    # 2. Contextual Nuance (The Regime)
    if structural_bias > 0.1 and detail_alignment < -0.1:
        label = "Structural Accumulation (The Dip)"
    elif structural_bias < -0.1 and detail_alignment > 0.1:
        label = "Tactical Distribution (The Bounce)"
    elif structural_bias > 0.2 and detail_alignment > 0.1:
        label = "Strong Bullish Momentum"
    elif structural_bias < -0.2 and detail_alignment < -0.1:
        label = "Strong Bearish Correction"
    else:
         label = base_label
         
    return label, total_score, stance_data

def forecast_spectral_path(trend_band, cycle_band, horizon=20):
    """
    Manifests a 'Conditional Future Path' by combining Structural Drift 
    from the trend component with Phase Extrapolation from the dominant cycle.
    """
    n = len(trend_band)
    recent_lookback = 30
    
    # 1. Structural Drift (Trend Extrapolation)
    t_series = trend_band[-recent_lookback:]
    x_t = np.arange(len(t_series))
    coeffs_t = np.polyfit(x_t, t_series, 1)
    
    future_x = np.arange(len(t_series), len(t_series) + horizon)
    trend_proj = np.polyval(coeffs_t, future_x)
    
    # 2. Cyclical Phase (Dominant Cycle Extrapolation)
    c_series = cycle_band[-recent_lookback:]
    x_c = np.arange(len(c_series))
    
    # Simple parabolic capture of local cycle curvature
    coeffs_c = np.polyfit(x_c, c_series, 2) 
    cycle_proj = np.polyval(coeffs_c, future_x)
    
    # 3. Aggregate Synthetic Path
    synthetic_path = trend_proj + cycle_proj
    
    return synthetic_path

def analyze_resonance(summary_data):
    """
    Interprets the lead-lag summary data into a semantic regime classification.
    """
    if not summary_data:
        return "Spectral Noise (Disconnected)", 0.0, "Decoupled"
        
    avg_coh = np.mean([r['avg_coherence'] for r in summary_data])
    # Weight lead by coherence
    leads = [r['lead_days'] for r in summary_data]
    cohs = [r['avg_coherence'] for r in summary_data]
    weighted_lead = np.average(leads, weights=cohs)
    
    # 1. Classification Logic
    if avg_coh < 0.35:
        regime = "Fragmented / Decoupled"
        desc = "Asset pair demonstrates high spectral entropy. No persistent lead-lag relationship detected."
    elif avg_coh > 0.65:
        if abs(weighted_lead) < 1.0:
            regime = "Harmonic Equilibrium (Sync)"
            desc = "Assets are in perfect phase lock. Strong systemic contagion; likely to respond to macro shocks in unison."
        elif weighted_lead > 1.0:
            regime = "Master-Slave Lead (Primary)"
            desc = "Primary asset exhibits dominant leadership across multiple cycles. Use as a reliable leading indicator."
        else:
            regime = "Master-Slave Lag (Primary)"
            desc = "Primary asset acts as a structural laggard. Watch the secondary asset for predictive signals."
    else:
        regime = "Emergent Resonance"
        desc = "Coherence is building but phase relationship remains volatile. Strategic caution advised."
        
    return regime, avg_coh, desc

def analyze_causality(cand_sym, target_sym, flow_delta, peak_flow, p_values):
    """
    Classifies the information flow relationship into semantic regimes.
    """
    # Use max p-value for a conservative check
    sig_confidence = 1 - np.min(list(p_values.values()))
    
    # 1. Classification Logic
    if peak_flow < 0.01:
        regime = "Stochastic Noise (No Flow)"
        desc = f"No meaningful information transfer detected between {cand_sym} and {target_sym}. Markets are likely decoupled."
    elif abs(flow_delta) > 0.05:
        if flow_delta > 0:
            regime = "Dominant Information Source"
            desc = f"{cand_sym} acts as a master oscillator, exerting structural influence over {target_sym}'s price action."
        else:
            regime = "Systemic Information Sink"
            desc = f"{cand_sym} is a passive respondent to information shocks originates from {target_sym}."
    elif sig_confidence > 0.95:
        regime = "Coupled Resonance (Feedback)"
        desc = "Asset pair exhibits strong bi-directional information flow. High systemic risk during regime shifts."
    else:
        regime = "Equilibrium Flow"
        desc = "Stable mutual dependency detected. No dominant leader identified at the current lag order."
        
    return regime, sig_confidence, desc

def analyze_backtest(results_full, results_oos):
    """
    Evaluates strategy performance and validation stability.
    """
    f_sharpe = results_full['sharpe']
    o_sharpe = results_oos['sharpe']
    f_trades = results_full['total_trades']
    
    # 1. Classification Logic
    if f_trades == 0:
        regime = "Zero-Signal Mismatch"
        desc = "The selected resonance parameters failed to generate any entry signals. TIP: Lower the 'Resonance Threshold' or select a higher-frequency band (e.g. 10-40 days) where lead/lag variance is higher."
    elif f_sharpe > 1.5 and o_sharpe > 1.0:
        regime = "Robust Resonance Capture"
        desc = "Strategy manifests significant alpha with stable out-of-sample persistence. High-fidelity spectral signal detected."
    elif f_sharpe > 0.5 and o_sharpe < 0:
        regime = "Spectral Decay (Overfit)"
        desc = "Significant performance divergence detected. Strategy likely captured in-sample noise rather than structural rhythms."
    elif f_sharpe < 0 and o_sharpe < 0:
        regime = "Antiphase Failure"
        desc = "Strategy is consistently inverse to price direction. May indicate a phase-alignment error or structural breakdown."
    else:
        regime = "Volatile / Speculative"
        desc = "Strategy demonstrates marginal alpha with low validation stability. High risk of capital erosion."
        
    validation_score = np.clip(o_sharpe / f_sharpe, -1.0, 1.0) if f_sharpe > 0.1 else 0.0
    
    return regime, validation_score, desc

def analyze_intraday(compression, dom_rhythm):
    """
    Interprets high-frequency spectral energy into semantic intraday regimes.
    """
    # Normalized score based on energy compression
    force_score = np.clip(compression / 5.0, 0.0, 1.0)
    
    # 1. Classification Logic
    if compression > 4.5:
        regime = "Impulsive Volatility Spike"
        desc = "Extreme energy concentration detected. The market is undergoing a structural breakout or reactive news event."
    elif compression > 2.8:
        regime = "Active Directional Search"
        desc = "Elevated spectral density suggests a transition phase. Rhythmic volatility is building toward a directional anchor."
    elif compression < 1.8:
        regime = "Spectral Entropy (Noise)"
        desc = "Energy is dispersed across all frequencies. No dominant intraday heartbeat detected; stay tactical/flat."
    else:
        regime = "Harmonic Equilibrium (Stable)"
        desc = "Intraday rhythms are locked and predictable. High-stability state suitable for mean-reversion or rhythm-following."
        
    return regime, force_score, desc

def analyze_portfolio(avg_resonance):
    """
    Evaluates global portfolio sync into semantic risk regimes.
    """
    # Force score mapping
    force_score = np.clip(avg_resonance / 0.8, 0.0, 1.0)
    
    # 1. Classification Logic
    if avg_resonance > 0.6:
        regime = "Systemic Contagion (Critical)"
        desc = "Extreme spectral alignment detected across the portfolio. Diversification is non-existent; the cluster behaves as a single levered bet."
    elif avg_resonance > 0.4:
        regime = "Harmonic Momentum Cluster"
        desc = "Significant synchronization detected. Assets are locked in a macro-regime; expect high downside correlation."
    elif avg_resonance < 0.2:
        regime = "Orthogonal Independence"
        desc = "High-fidelity diversification achieved. Localized shocks are likely to be isolated, providing robust risk masking."
    else:
        regime = "Standard Diversified Mix"
        desc = "Portfolio resonance is within institutional bounds. Reasonable balance between coordination and independence."
        
    return regime, force_score, desc
def get_execution_playbook(module, regime):
    """
    Provides tactical instructions based on the detected regime.
    """
    playbooks = {
        "Decomposition": {
            "Strong Bullish Momentum": [
                "1. Confirm 'Trend Line' slope is positive.",
                "2. Look for price pullbacks toward the 'Underlying Structural Trend' for entry.",
                "3. Stop Loss: Set below the recent 40-day cycle trough."
            ],
            "Structural Accumulation (The Dip)": [
                "1. Observe 'Weekly/Monthly' cycles turning positive.",
                "2. Establish long core positions while structural bias remains bullish.",
                "3. Exit: Fade the move when short-term cycles peak above price."
            ],
            "Neutral / Consolidating": [
                "1. Deploy mean-reversion strategies.",
                "2. Sell the cycle peaks and buy the troughs.",
                "3. Caution: Breakout may be imminent if energy compression builds."
            ]
        },
        "Coherence": {
            "Master-Slave Lead (Primary)": [
                "1. Set 'Traded Asset' to the laggard.",
                "2. Wait for the 'Signal Asset' to move by >1.5 StdDev.",
                "3. Execute trade in laggard assuming 1-5 day lead preservation."
            ],
            "Harmonic Equilibrium (Sync)": [
                "1. Treat assets as a single macro-lever.",
                "2. Avoid using either as a lead/lag signal.",
                "3. Strategy: Pair-trading or market-neutral hedging."
            ],
            "Fragmented / Decoupled": [
                "1. Halt all cross-spectral strategies.",
                "2. Assets are moving on idiosyncratic news.",
                "3. Monitor for emergent 'Resonance Hubs' before re-entering."
            ]
        },
        "Causality": {
            "Dominant Information Source": [
                "1. This asset is the 'Master Oscillator'.",
                "2. Trust its signal direction over the target asset's local price action.",
                "3. High-Fidelity Signal: Confirm flow delta is > 0.1 at short lags."
            ],
            "Coupled Resonance (Feedback)": [
                "1. High Risk Zone. Feedback loops create 'Black Swan' potential.",
                "2. Avoid directional bets based on lead/lag.",
                "3. Strategy: Delta-neutral vol-selling if peak flow is stable."
            ]
        },
        "Backtesting": {
            "Robust Resonance Capture": [
                "1. Scaling: Use Kelly Fraction for position sizing.",
                "2. Monitoring: Watch 'OS Sharpe' for performance decay.",
                "3. Risk: Set trailing stops at the 20-day cycle standard deviation."
            ],
            "Spectral Decay (Overfit)": [
                "1. Strategy Rejection: In-sample alpha is illusory.",
                "2. Action: Re-tune to a slower frequency band (higher periodicity).",
                "3. Optimization: Increase 'Phase Smoothing' to filter noise spikes."
            ]
        },
        "Intraday": {
            "Impulsive Volatility Spike": [
                "1. Breakout Strategy: Trade the momentum direction.",
                "2. Time Horizon: 60-120 minutes (Short Burst).",
                "3. Caution: High probability of 'Mean Reversion' once energy dissipates."
            ],
            "Harmonic Equilibrium (Stable)": [
                "1. Oscillation Strategy: Fade the range extremes.",
                "2. Signal: Entry when STFT energy dips below the alpha-threshold.",
                "3. Exit: Close at the 'Dominant Intraday Rhythm' center-line."
            ]
        },
        "Portfolio": {
            "Systemic Contagion (Critical)": [
                "1. Liquidation/Hedge Alert: Diversification is broken.",
                "2. Action: Reduce gross exposure by 30-50%.",
                "3. Solution: Inject 'Orthogonal Node' (e.g. GLD or BTC if decoupled)."
            ],
            "Orthogonal Independence": [
                "1. Structural Safety: Portfolio is well-diversified.",
                "2. Action: Maintain core allocations.",
                "3. Optimization: Rebalance based on individual asset 'Resonance Hubs'."
            ]
        }
    }
    
    # Return default if specific regime not mapped
    module_guide = playbooks.get(module, {})
    return module_guide.get(regime, ["1. Monitor spectral stability.", "2. Wait for clear regime transition.", "3. Stay tactical/flat in high-entropy states."])
