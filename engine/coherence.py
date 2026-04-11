import numpy as np
import pandas as pd
import pycwt as wavelet


def calculate_coherence(first_series, second_series, time_step=1.0, scale_resolution=1/12):
    """
    Measures how much two signals "resonate" with each other at different
    frequencies and points in time.

    Returns:
        coherence_map  — (n_freqs × n_times), values in [0, 1]
        phase_angle    — (n_freqs × n_times), radians; positive = first series leads
        coi            — cone of influence (boundary reliability limit)
        freqs          — frequency array (cycles per time_step unit)
        sig            — significance mask (None when sig=False, kept for API compat)

    Phase convention:
        phase > 0  →  first_series leads second_series at that frequency/time
        phase < 0  →  second_series leads first_series
        |lead in days| = |phase| / (2π × freq)
    """
    standard_wave = wavelet.Morlet()

    wavelet.xwt(
        first_series, second_series, time_step,
        dj=scale_resolution, wavelet=standard_wave,
    )

    coherence_map, phase_angle, coi, freqs, sig = wavelet.wct(
        first_series, second_series, time_step,
        dj=scale_resolution, wavelet=standard_wave, sig=False,
    )

    return coherence_map, phase_angle, coi, freqs, sig


def compute_lead_lag_summary(phase_angle, freqs, coherence_map, coi,
                              min_coherence=0.5, n_time=None):
    """
    Converts the raw phase angle map into an actionable lead/lag summary.

    For each frequency band (inside the cone of influence and above a minimum
    coherence threshold) this returns:
      - period_days   : cycle period in trading days (1 / freq)
      - lead_days     : how many days the first series leads the second;
                        positive = first leads, negative = second leads
      - first_leads   : True when first_series is ahead at this frequency
      - avg_coherence : average coherence inside COI at this frequency

    Rows are sorted by period (shortest cycle first) and only bands with
    meaningful coherence are included.

    Typical use: pass first_sym returns as first_series and second_sym as second.
    If lead_days > 0 for the 20-day band, first_sym moves first at the monthly
    cycle — use it as a leading indicator for second_sym.
    """
    n_freqs = len(freqs)
    rows = []

    for fi in range(n_freqs):
        freq = freqs[fi]
        if freq <= 0:
            continue

        period = 1.0 / freq

        # Mask to inside the cone of influence at this scale
        inside_coi = period <= coi  # coi is in the same units as period

        if not np.any(inside_coi):
            continue

        coh_inside = coherence_map[fi, inside_coi]
        phase_inside = phase_angle[fi, inside_coi]

        if np.mean(coh_inside) < min_coherence:
            continue

        # Weight phase average by coherence so high-coherence windows dominate
        weights = np.clip(coh_inside, 1e-6, None)
        avg_phase = float(np.average(phase_inside, weights=weights))

        # Lead time: how many days does the first series precede the second
        lead_days = avg_phase * period / (2.0 * np.pi)

        rows.append({
            "period_days": round(period, 1),
            "freq": round(freq, 4),
            "lead_days": round(lead_days, 1),
            "first_leads": bool(lead_days > 0),
            "avg_coherence": round(float(np.mean(coh_inside)), 3),
            "avg_phase_deg": round(float(np.degrees(avg_phase)), 1),
        })

    return sorted(rows, key=lambda r: r["period_days"])


def check_coherence_significance(a, b, dt=1.0, dj=1/12):
    """
    Checks if the observed resonance is statistically significant
    or just random noise (Monte Carlo method).
    """
    wave = wavelet.Morlet()
    return wavelet.wct_significance(a, b, dt, dj=dj, mother=wave)
