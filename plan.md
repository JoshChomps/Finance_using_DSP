# FinSignal Suite — AlgoFest 2026 Project Plan

## Executive Summary

**Project Name:** FinSignal Suite — Multi-Resolution Financial Signal Processing Engine  
**Hackathon:** AlgoFest Hackathon 2026: Battle of the Beasts  
**Track:** Best FinTech Solution  
**Deadline:** April 29, 2026  
**Time Available:** ~20 days from April 9  

**One-Liner:** A real-time multi-resolution spectral analysis engine that decomposes financial asset volatility into frequency bands, detects cross-asset resonance via wavelet coherence, and reveals directional causal coupling — things traditional correlation metrics completely miss.

---

## Part 1: Problem Statement & Innovation Framing

### The Problem

Traditional portfolio risk analysis relies on correlation metrics (Pearson, rolling-window) that treat all timescales identically. This creates a dangerous blind spot: two assets can appear uncorrelated over daily returns while being tightly coupled at monthly or quarterly frequencies — exactly the timescales that matter for institutional allocation decisions. Fund managers making diversification decisions on flat correlation numbers are unknowingly exposed to frequency-dependent risk that only becomes visible during regime changes (crises, rate pivots, sector rotations).

Furthermore, standard correlation is symmetric — it cannot tell you *which* asset leads and which follows, or whether that leadership relationship changes over time and across frequencies.

### Our Approach

Apply multi-resolution spectral analysis (wavelet transforms, STFT, DFT) to financial time series to:

1. **Decompose** asset returns into distinct frequency bands (intraday noise, weekly cycles, monthly trends, macro regime shifts)
2. **Detect resonance** between asset pairs using wavelet coherence — showing *at which frequencies* and *during which time periods* assets are coupled
3. **Infer directionality** using spectral Granger causality — revealing which asset *leads* at each frequency band

### Why This Is Novel at a Hackathon

- Nobody applies DSP to finance at hackathons — this isn't "I called an API," it's "I implemented a multi-resolution spectral analysis engine from first principles"
- The core algorithms (wavelet transforms, cross-spectral density, spectral Granger causality) are mathematically non-trivial
- The insight it produces (frequency-dependent correlation, directional coupling) is genuinely useful to professional quant traders
- The demo is visceral — you can *see* hidden structure emerge when you decompose noisy price data

---

## Part 2: Judging Criteria Alignment

AlgoFest evaluates on these criteria. Here's how we map to each:

### Algorithmic Innovation (Primary Weight)
- **Wavelet decomposition** of financial time series using both DWT (for discrete band separation) and CWT (for continuous scalograms)
- **Synchrosqueezing transform** (via ssqueezepy) — an advanced technique that sharpens time-frequency resolution beyond standard CWT, demonstrating cutting-edge DSP knowledge
- **Wavelet coherence** with phase arrows — not just "are these correlated?" but "at what frequency, when, and who leads?"
- **Spectral Granger causality** — causal inference in the frequency domain, a technique from computational neuroscience applied to finance
- Everything is implemented from signal processing fundamentals, not wrapped API calls

### Technical Implementation
- Clean, modular Python codebase with clear separation of engine (algorithms), data (ingestion), and UI (dashboard)
- Proper use of vectorized numpy/scipy operations for performance
- Caching strategy for expensive wavelet computations
- Well-documented README with architecture diagrams, algorithm explanations, and setup instructions

### Usability and User Experience
- Interactive Streamlit dashboard with sensible defaults
- Pre-loaded example (SPY vs QQQ) so judges can explore immediately
- Tooltips explaining each visualization for non-DSP audiences
- Slider-based parameter exploration (wavelet family, decomposition depth, frequency band selection)

### Scalability and Feasibility
- Stateless computation pipeline — horizontally scalable
- All operations are vectorized numpy/scipy — handles thousands of tickers
- Data layer is provider-agnostic (can swap yfinance for Finnhub WebSocket for production streaming)
- Architecture designed for eventual deployment behind a REST API

### Clear Problem Identification
- Frame around the specific failure mode of traditional correlation metrics
- Reference real-world examples (2008 crisis: assets that appeared uncorrelated were actually coupled at low frequencies)
- Quantify the gap: "Pearson correlation between SPY and GLD is 0.05 — but at the 6-month frequency band, their coherence is 0.82"

---

## Part 3: Data Layer — API Evaluation

### Decision: Dual-Source Strategy

**Primary (Historical/Bulk):** yfinance  
**Secondary (Real-Time Demo):** Finnhub Free Tier  

### Evaluation Matrix

| Provider | Free Tier | Historical Depth | Real-Time | WebSocket | Rate Limit | Verdict |
|----------|-----------|-----------------|-----------|-----------|------------|---------|
| **yfinance** | Unlimited (scraper) | 20+ years daily, 7 days intraday | Near-real-time (unofficial) | No | Informal, can be blocked | Best for bulk historical. Fragile but fine for hackathon. |
| **Finnhub** | 60 calls/min | Limited on free | Yes (US markets) | Yes (50 symbols free) | 60/min, 30/sec | Best free real-time. WebSocket is rare at $0. |
| **Alpha Vantage** | 25 calls/day | Good | 15-min delay on free | No | 25/day — unusable for demo | Too restrictive. |
| **Polygon.io** | 5 calls/min | Limited on free | Paid only at production quality | Paid | 5/min | Too expensive for our needs. |
| **FMP** | 250 calls/day | Good | Paid tiers | Paid | 250/day | Middle ground but no free WebSocket. |

### Why This Dual Strategy

- **yfinance** gives us unlimited access to 20+ years of daily OHLCV data for any ticker — perfect for the wavelet decomposition demo where long history = more frequency resolution at low bands. Its fragility (scraping Yahoo's backend) is acceptable for a hackathon because we can pre-cache data locally as CSV fallback.
- **Finnhub** gives us the "live" demo moment. Its free WebSocket feed (up to 50 symbols) lets us show real-time trade data flowing into the pipeline. Even if it's just US equities, that's enough for the demo. 60 API calls/minute is generous for fetching supplementary data.
- **Fallback plan:** Pre-download 5 years of daily data for 20 representative tickers (SPY, QQQ, GLD, TLT, AAPL, MSFT, XLE, XLF, etc.) and store as local CSVs. If any API breaks during the live demo, we switch to cached data seamlessly.

### Data Requirements

| Use Case | Source | Frequency | History Needed |
|----------|--------|-----------|---------------|
| Wavelet decomposition | yfinance | Daily close | 3-5 years (750-1250 points) |
| CWT scalogram | yfinance | Daily close | 2+ years |
| Cross-asset coherence | yfinance | Daily close | 3+ years (both assets) |
| Real-time streaming demo | Finnhub WebSocket | Tick/trade | Live |
| Intraday spectral demo | yfinance | 1-hour bars | 7 days (max free intraday) |

---

## Part 4: Signal Processing Engine — Library & Algorithm Decisions

This is the core of the project. Every library choice here was evaluated for correctness, performance, and algorithmic depth.

### Library Selection

#### PyWavelets (pywt) — Discrete Wavelet Transform

**Role:** DWT decomposition of returns into frequency bands, inverse DWT for band reconstruction.

**Why pywt:** It's the industry-standard Python wavelet library, battle-tested, and supports every wavelet family we need (Daubechies, Symlets, Coiflets, biorthogonal). It has rock-solid DWT/IDWT with proper boundary handling. Its MODWT (maximal overlap DWT) implementation avoids the shift-variance problem of standard DWT, which matters for financial time series where alignment is critical.

**What it lacks:** Its CWT implementation has known issues (acknowledged in their own docs — they recommend ssqueezepy for CWT). It has no built-in wavelet coherence.

**We use it for:** DWT multi-level decomposition, band-specific reconstruction, and discrete frequency-band volatility measurement.

#### PyCWT (pycwt) — Cross-Wavelet Transform & Coherence

**Role:** Wavelet coherence between two time series, cross-wavelet transform, phase relationship extraction.

**Why pycwt:** This is the only mature Python library that implements *wavelet coherence* out of the box, including:
- Cross-wavelet transform (XWT) with significance testing
- Wavelet transform coherence (WTC) — the frequency-localized equivalent of correlation
- Phase arrows showing lead/lag relationships
- Cone of influence masking to avoid edge effects
- Statistical significance testing against red noise backgrounds

It's based on the Torrence & Compo (1998) wavelet analysis framework and the Grinsted et al. (2004) cross-wavelet/coherence methodology — these are the canonical references in the field.

**What it lacks:** It only does Morlet wavelets for CWT (which is actually the right choice for coherence analysis — Morlet gives the best time-frequency tradeoff for oscillatory signals).

**We use it for:** The entire cross-asset resonance detection module. This is Module B — the "coupled oscillator" analysis.

#### ssqueezepy — Synchrosqueezed Wavelet Transform

**Role:** High-resolution CWT scalograms with synchrosqueezing for sharper time-frequency representation.

**Why ssqueezepy:** This is the most advanced time-frequency analysis library in Python. It implements:
- CWT that is substantially faster than pywt's CWT (benchmarked 10-50x faster)
- **Synchrosqueezing** — a technique that concentrates the wavelet coefficients to give sharper frequency localization than standard CWT. This is a genuine algorithmic innovation (Daubechies & Maes, 1996) that demonstrates advanced DSP knowledge.
- Multi-threaded execution by default
- GPU acceleration support (CuPy/PyTorch) — impressive for scalability claims even if we don't use it in the demo

**What it lacks:** No cross-wavelet/coherence (that's pycwt's job).

**We use it for:** Generating the beautiful CWT scalograms and synchrosqueezed scalograms in the decomposition panel. The side-by-side comparison of standard CWT vs synchrosqueezed CWT is a visual "wow" moment that demonstrates algorithmic sophistication.

#### scipy.signal — Complementary DSP

**Role:** STFT spectrograms, Butterworth bandpass filters, Welch PSD estimation, coherence function.

**Why scipy.signal:** Standard library, no extra dependencies. We use it for:
- `scipy.signal.stft` — Short-Time Fourier Transform for comparison with wavelet approaches
- `scipy.signal.butter` + `sosfiltfilt` — Butterworth bandpass filters for clean band separation
- `scipy.signal.welch` — Welch's method for power spectral density estimation
- `scipy.signal.coherence` — Fourier-domain coherence as a baseline comparison to wavelet coherence
- `scipy.fft` — Raw FFT/PSD for the power spectrum visualization

#### statsmodels — Granger Causality

**Role:** Time-domain Granger causality as a baseline, with VAR model fitting.

**Why statsmodels:** `statsmodels.tsa.stattools.grangercausalitytests` gives us a well-tested, publication-quality implementation of standard Granger causality with proper statistical testing (F-test, chi-squared). We use this as the baseline before upgrading to frequency-domain Granger causality.

#### Custom Implementation — Spectral Granger Causality

**Role:** Frequency-domain Granger causality showing *at which frequencies* one asset Granger-causes another.

**Why custom:** No mature Python library does this out of the box. We implement it using the Geweke (1982) spectral decomposition of Granger causality:

1. Fit a bivariate VAR model to the two return series
2. Compute the transfer function H(f) and spectral matrix S(f) from the VAR coefficients
3. Decompose the spectral matrix to get frequency-specific Granger causality measures: Ix→y(f), Iy→x(f), and Ixy(f) (instantaneous causality)

This is algorithmically the most impressive component — it's causal inference in the frequency domain, borrowed from computational neuroscience (where it's used to study brain connectivity). The pyGC library provides reference implementations we can study, but we'll write our own clean version for the codebase.

**Alternative considered:** Transfer entropy (PyCausality package). Decided against it because:
- Transfer entropy requires careful binning/kernel density estimation choices that are hard to get right
- Spectral Granger causality is more directly interpretable alongside our wavelet coherence results
- Both are equivalent for Gaussian variables (Barnett et al., 2009), and financial returns are approximately Gaussian after log transformation

### Algorithm Pipeline Summary

```
Raw Price Data
    │
    ├── Log Returns Computation
    │
    ├── MODULE A: Decomposition
    │   ├── DWT Multi-Level Decomposition (pywt)
    │   │   └── Band reconstruction → volatility per band
    │   ├── CWT Scalogram (ssqueezepy)
    │   │   └── Synchrosqueezed variant for comparison
    │   ├── STFT Spectrogram (scipy.signal)
    │   └── Welch PSD (scipy.signal)
    │
    ├── MODULE B: Cross-Asset Coherence
    │   ├── Wavelet Coherence + Phase (pycwt)
    │   ├── Fourier Coherence baseline (scipy.signal)
    │   └── Rolling windowed correlation (pandas) as naive baseline
    │
    └── MODULE C: Directional Causality
        ├── Time-domain Granger causality (statsmodels)
        └── Spectral Granger causality (custom, Geweke 1982)
```

### Wavelet Family Selection

For financial time series, the wavelet choice matters:

| Wavelet | Best For | We Use It In |
|---------|----------|-------------|
| **Morlet** | CWT coherence analysis (best time-frequency tradeoff for oscillatory content) | pycwt coherence, ssqueezepy scalograms |
| **Daubechies db4/db8** | DWT decomposition (compact support, good frequency separation) | pywt DWT band decomposition |
| **Symlet sym8** | DWT when near-symmetry matters (less phase distortion) | Alternative DWT option in UI |

---

## Part 5: Dashboard & Visualization — Framework Decision

### Decision: Streamlit

### Why Streamlit Over Alternatives

| Framework | Speed to Build | Visual Quality | Interactivity | Deployment | Verdict |
|-----------|---------------|---------------|---------------|------------|---------|
| **Streamlit** | Fastest (script → app) | Good (Plotly integration) | Widget-based, sufficient | Streamlit Cloud (free, 1-click) | **Winner for our timeline** |
| **Dash (Plotly)** | Moderate (callback wiring) | Best (native Plotly) | Full callback control | Heroku/Render | Better for production, overkill for 20 days |
| **Panel** | Moderate | Good | Reactive, powerful | Harder to deploy free | More flexible but steeper learning curve |
| **Gradio** | Fast | Limited | ML-demo oriented | HuggingFace Spaces | Wrong tool for financial dashboards |

**Critical factors:**
- Streamlit Community Cloud is free, deploys directly from a GitHub repo, and the "live demo link" requirement is satisfied instantly
- Streamlit's rerun-on-widget-change model is actually perfect for our use case: user changes a parameter → recompute decomposition → redraw plots
- Plotly charts inside Streamlit give us interactive zoom, hover tooltips, and pan — exactly what judges need to explore the data
- Streamlit's `st.cache_data` decorator handles expensive wavelet computations elegantly

**Limitation awareness:** Streamlit Community Cloud has limited CPU/memory. Our mitigation:
- Pre-compute decompositions for the default demo tickers and cache results
- Limit CWT scale range to prevent memory blowout on free tier
- Use `@st.cache_data` aggressively
- Store pre-computed demo data as parquet files in the repo

### Dashboard Panels

**Panel 1 — Decomposition Explorer**
- Ticker selector (dropdown, default: SPY)
- Wavelet family selector (Daubechies, Symlets, Morlet)
- Decomposition depth slider (3-8 levels)
- Main chart: original price with overlay toggle for each frequency band
- Sub-chart: power at each band as a stacked area chart
- Optional: side-by-side CWT scalogram vs synchrosqueezed scalogram

**Panel 2 — Cross-Asset Resonance**
- Two-ticker selector (defaults: SPY + GLD)
- Wavelet coherence heatmap (time × frequency × coherence magnitude)
- Phase arrow overlay showing lead/lag
- Band-by-band correlation comparison (bar chart: Pearson vs wavelet coherence per band)
- Highlighted "resonance events" where coherence spikes above threshold

**Panel 3 — Directional Causality**
- Two-ticker selector
- Spectral Granger causality plot: Ix→y(f) and Iy→x(f) as functions of frequency
- Time-domain Granger causality summary (p-values table)
- Interpretation card: "At weekly frequencies, AAPL leads MSFT (GC = 0.34). At monthly frequencies, the relationship reverses (GC = 0.18 vs 0.42)."

**Panel 4 — Real-Time Feed (Stretch Goal)**
- Finnhub WebSocket connection to live trade data
- Rolling 1-hour STFT spectrogram updating in near-real-time
- Simple visual — this is a "bonus" that shows the pipeline can handle streaming

### Visualization Library: Plotly

**Why Plotly over Matplotlib:**
- Interactive by default (zoom, pan, hover) — critical for judges exploring data
- Heatmaps with hover values (for coherence maps)
- Subplots with shared axes (for time-aligned decomposition bands)
- Native Streamlit integration via `st.plotly_chart`
- Professional appearance without extensive styling

**One exception:** For the CWT scalogram with phase arrows, we may need a Matplotlib figure rendered as an image (pycwt's built-in plotting uses Matplotlib and drawing arrows on a Plotly heatmap is cumbersome). This is acceptable — render as a static image within Streamlit.

---

## Part 6: Deployment

### Primary: Streamlit Community Cloud

- **Cost:** Free
- **Setup:** Connect GitHub repo → auto-deploy on push
- **URL:** https://finsignal-suite.streamlit.app (or similar)
- **Limitations:** Limited CPU/memory, apps sleep after inactivity, public repo required
- **Mitigation:** Pre-cache demo data, aggressive caching, lightweight default parameters

### Fallback: Render Free Tier

- If Streamlit Cloud's resources are too constrained for wavelet computations
- Render offers 750 free hours/month of web service hosting
- Docker-based deployment, slightly more setup

### Repository: GitHub (Public)

Required by both Devpost submission and Streamlit Cloud deployment. Structure:

```
finsignal-suite/
├── app.py                          # Streamlit entry point
├── pages/
│   ├── 1_Decomposition.py          # Panel 1
│   ├── 2_Coherence.py              # Panel 2
│   ├── 3_Causality.py              # Panel 3
│   └── 4_Live_Feed.py              # Panel 4 (stretch)
├── engine/
│   ├── __init__.py
│   ├── data.py                     # Data fetching, caching, normalization
│   ├── decompose.py                # DWT decomposition, band reconstruction
│   ├── scalogram.py                # CWT, synchrosqueezed CWT, STFT
│   ├── spectrum.py                 # FFT, PSD, Welch
│   ├── coherence.py                # Wavelet coherence (pycwt wrapper)
│   ├── granger.py                  # Time-domain & spectral Granger causality
│   └── utils.py                    # Log returns, normalization, windowing
├── data/
│   └── cache/                      # Pre-computed demo data (parquet)
├── tests/
│   ├── test_decompose.py
│   ├── test_coherence.py
│   └── test_granger.py
├── docs/
│   ├── architecture.md
│   └── algorithms.md               # Mathematical documentation
├── .streamlit/
│   └── config.toml                 # Theme, layout settings
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Part 7: Requirements & Dependencies

```
# requirements.txt
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
PyWavelets>=1.5.0
pycwt>=0.4.0b0
ssqueezepy>=0.6.6
plotly>=5.18.0
yfinance>=0.2.36
statsmodels>=0.14.0
finnhub-python>=2.4.19
pyarrow>=14.0.0            # For parquet caching
```

**Total estimated install size:** ~300MB (numpy/scipy dominate)  
**Python version:** 3.11+ (required by ssqueezepy for some features)

---

## Part 8: 20-Day Build Plan

### Phase 1: Foundation (Days 1-4, April 9-12)

**Day 1 — Data Layer & Validation**
- [ ] Set up repo, virtual environment, install all dependencies
- [ ] Implement `engine/data.py`: yfinance fetcher with local CSV fallback
- [ ] Download and cache 5-year daily data for 20 tickers
- [ ] Verify data quality: check for gaps, splits, dividends
- [ ] Write `engine/utils.py`: log returns, normalization, windowing functions

**Day 2 — DWT Decomposition Core**
- [ ] Implement `engine/decompose.py`: multi-level DWT with pywt
- [ ] Test on SPY: decompose 5 years of daily returns into 6 bands
- [ ] Implement band reconstruction (IDWT per level)
- [ ] Verify energy conservation: sum of band energies ≈ total signal energy
- [ ] Notebook exploration: visualize each band, confirm frequency interpretation

**Day 3 — CWT & Scalograms**
- [ ] Implement `engine/scalogram.py`: CWT with ssqueezepy
- [ ] Generate standard CWT scalogram for SPY
- [ ] Generate synchrosqueezed CWT scalogram — compare sharpness
- [ ] Implement STFT spectrogram as comparison view
- [ ] Add Welch PSD computation

**Day 4 — First Streamlit Page**
- [ ] Create `app.py` and `pages/1_Decomposition.py`
- [ ] Wire up ticker selector → data fetch → DWT decomposition → Plotly charts
- [ ] Add wavelet family and depth sliders
- [ ] Implement `@st.cache_data` for expensive computations
- [ ] **Milestone:** Working decomposition panel with interactive controls

### Phase 2: Cross-Asset Analysis (Days 5-9, April 13-17)

**Day 5 — Wavelet Coherence**
- [ ] Implement `engine/coherence.py`: pycwt wavelet coherence wrapper
- [ ] Test on SPY vs QQQ (should show high coherence at all frequencies)
- [ ] Test on SPY vs GLD (should show low high-frequency, higher low-frequency coherence)
- [ ] Extract phase arrows from pycwt output
- [ ] Handle cone-of-influence masking

**Day 6 — Coherence Visualization**
- [ ] Build coherence heatmap in Plotly (time × period × coherence)
- [ ] Add phase arrow overlay (likely Matplotlib rendered as image)
- [ ] Build band-by-band comparison chart (Pearson vs wavelet coherence per band)
- [ ] Add significance contours (95% confidence against red noise)

**Day 7 — Coherence Dashboard Panel**
- [ ] Create `pages/2_Coherence.py`
- [ ] Wire up two-ticker selector → coherence computation → visualization
- [ ] Add "resonance event" detection: flag periods where coherence > 0.8
- [ ] Add interpretation text cards
- [ ] **Milestone:** Working coherence panel showing frequency-dependent correlation

**Day 8 — Granger Causality (Time Domain)**
- [ ] Implement `engine/granger.py`: statsmodels Granger causality wrapper
- [ ] Test on known causal pairs (e.g., oil price → airline stocks)
- [ ] Display results as p-value table with lag selection

**Day 9 — Spectral Granger Causality**
- [ ] Implement spectral Granger causality (Geweke 1982 decomposition)
- [ ] Fit bivariate VAR model
- [ ] Compute transfer function H(f) and spectral matrix S(f)
- [ ] Decompose into Ix→y(f), Iy→x(f), Ixy(f)
- [ ] Validate: integral of spectral GC over all frequencies should equal time-domain GC
- [ ] **Milestone:** Can show *at which frequencies* one asset Granger-causes another

### Phase 3: Integration & Polish (Days 10-14, April 18-22)

**Day 10 — Causality Dashboard Panel**
- [ ] Create `pages/3_Causality.py`
- [ ] Wire up spectral GC visualization (two lines: x→y and y→x vs frequency)
- [ ] Add time-domain GC comparison
- [ ] Add interpretation cards: "At weekly frequencies, AAPL leads MSFT"

**Day 11 — UX Polish**
- [ ] Add landing page (app.py) with project overview and guided walkthrough
- [ ] Pre-loaded demo mode with SPY + QQQ + GLD
- [ ] Tooltips on every visualization explaining what it shows
- [ ] Consistent color scheme across all panels
- [ ] Loading spinners for expensive computations

**Day 12 — Performance Optimization**
- [ ] Profile bottlenecks (likely CWT computation and coherence)
- [ ] Pre-compute and cache demo ticker results as parquet
- [ ] Optimize scale ranges for CWT (limit to meaningful frequency bands)
- [ ] Test on Streamlit Cloud — verify it doesn't OOM
- [ ] Add graceful error handling for API failures

**Day 13 — Real-Time Feed (Stretch Goal)**
- [ ] Implement Finnhub WebSocket connection
- [ ] Rolling STFT spectrogram on live tick data
- [ ] Create `pages/4_Live_Feed.py`
- [ ] If this doesn't work well on Streamlit Cloud, demote to local-only demo

**Day 14 — Testing & Edge Cases**
- [ ] Write tests for core engine functions
- [ ] Test with different asset classes: stocks, ETFs, crypto (via yfinance)
- [ ] Test edge cases: very short time series, flat/zero returns, missing data
- [ ] Cross-browser testing of Streamlit dashboard

### Phase 4: Documentation & Submission (Days 15-20, April 23-29)

**Day 15 — README & Technical Documentation**
- [ ] Write comprehensive README with architecture diagram (Mermaid)
- [ ] Setup instructions: clone → pip install → streamlit run
- [ ] Algorithm documentation: mathematical formulations with references
- [ ] Screenshots of each dashboard panel

**Day 16 — Architecture Diagram**
- [ ] Create Mermaid flowchart for README
- [ ] Create system architecture diagram for presentation deck
- [ ] Document data flow: API → engine → cache → dashboard

**Day 17 — Demo Video Script & Recording**
- [ ] Script the 2-3 minute demo video (see Demo Script below)
- [ ] Record screen capture with voiceover
- [ ] Edit: title card, transitions, captions for key moments

**Day 18 — Presentation Deck (Optional but Recommended)**
- [ ] Problem slide: "Correlation lies to you"
- [ ] Solution slide: multi-resolution spectral analysis
- [ ] Architecture slide: system diagram
- [ ] Demo screenshots with annotations
- [ ] Impact slide: who uses this and why
- [ ] Technical depth slide: algorithm summary with equations

**Day 19 — Deploy & Final Testing**
- [ ] Deploy to Streamlit Cloud
- [ ] Verify live demo link works
- [ ] Test from a different device/network
- [ ] Final bug fixes

**Day 20 — Devpost Submission (April 29)**
- [ ] Fill out all Devpost fields
- [ ] Upload demo video
- [ ] Link GitHub repo (ensure public)
- [ ] Link live demo
- [ ] List all technologies used
- [ ] Team details
- [ ] Final review of everything

---

## Part 9: Demo Script (2-3 Minutes)

### Opening (0:00 - 0:20)
"Every fund manager uses correlation to measure portfolio risk. But correlation is a lie — it hides frequency-dependent structure that only becomes visible during market crises. FinSignal Suite uses multi-resolution spectral analysis to decompose what correlation cannot."

### Decomposition Demo (0:20 - 1:00)
- Show SPY daily returns — looks like random noise
- Click decompose → 6 frequency bands appear, overlaid on the original
- "The lowest band captures macro regime shifts. The mid bands capture weekly and monthly cycles. The highest bands are pure noise."
- Toggle bands on/off: "Watch how the mid-frequency band alone captures the COVID crash and recovery perfectly"

### Coherence Demo (1:00 - 1:50)
- Select SPY and GLD
- "Pearson correlation between these is 0.05 — you'd think they're independent"
- Show wavelet coherence heatmap
- "But look at the 3-6 month frequency band: coherence is 0.75 during 2020-2021. At monthly and quarterly horizons, these assets are tightly coupled — exactly when diversification is supposed to protect you."
- Show phase arrows: "And gold *leads* — the coupling is directional"

### Causality Demo (1:50 - 2:30)
- Switch to spectral Granger causality view
- "At high frequencies, neither asset Granger-causes the other. But at low frequencies, gold→stocks causality is significant while stocks→gold is not."
- "This means gold price movements at monthly timescales *predict* subsequent stock movements — a signal invisible to traditional analysis."

### Closing (2:30 - 2:50)
"FinSignal Suite gives portfolio managers a new lens: not just *if* assets are related, but *at which frequencies*, *when*, and *who leads*. Built from signal processing fundamentals — wavelet transforms, spectral analysis, and frequency-domain causal inference."

---

## Part 10: Risk Assessment & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| yfinance API breaks | Medium | High | Pre-cached CSV data for all demo tickers |
| Streamlit Cloud OOM on CWT | Medium | High | Limit scale range, pre-compute results, use parquet cache |
| Spectral Granger implementation too complex | Medium | Medium | Fall back to time-domain Granger at multiple frequency bands (pre-filter → Granger per band) |
| pycwt/ssqueezepy installation issues on Streamlit Cloud | Low | High | Pin versions, test deployment early (Day 4) |
| Finnhub WebSocket unreliable in Streamlit | High | Low | This is a stretch goal — degrade gracefully to replay mode |
| Demo video quality | Low | Medium | Write script in advance, record multiple takes |
| Judges don't understand DSP | Medium | Medium | Interpretation cards on every panel, clear "so what" framing |

---

## Part 11: Post-Hackathon Vision

This is designed to outlive the hackathon:

- **Add more assets:** Crypto (high volatility = rich frequency content), forex, commodities
- **Add a backtesting module:** "If you traded the coherence signal (buy when cross-asset coherence drops below 0.3, sell when it rises above 0.7), what's the Sharpe ratio?"
- **Production streaming:** Replace yfinance with Polygon.io paid tier + Kafka for real-time streaming pipelines
- **REST API:** Wrap the engine in FastAPI, expose decomposition and coherence as API endpoints
- **Research paper potential:** The combination of wavelet coherence + spectral Granger causality applied to financial portfolio construction is a publishable contribution

---

## Part 12: Key References

1. **Torrence, C. & Compo, G.P.** (1998). A Practical Guide to Wavelet Analysis. *Bulletin of the American Meteorological Society*, 79, 61-78. — *Canonical wavelet analysis reference*
2. **Grinsted, A., Moore, J.C. & Jevrejeva, S.** (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear Processes in Geophysics*, 11, 561-566. — *Cross-wavelet and coherence methodology*
3. **Geweke, J.** (1982). Measurement of linear dependence and feedback between multiple time series. *Journal of the American Statistical Association*, 77, 304-313. — *Spectral Granger causality decomposition*
4. **Daubechies, I. & Maes, S.** (1996). A Nonlinear squeezing of the Continuous Wavelet Transform. — *Synchrosqueezing foundation*
5. **Mallat, S.** (2008). *A Wavelet Tour of Signal Processing: The Sparse Way*. Academic Press. — *Comprehensive wavelet theory*
6. **Barnett, L., Barrett, A.B. & Seth, A.K.** (2009). Granger causality and transfer entropy are equivalent for Gaussian variables. *Physical Review Letters*, 85(2). — *Justification for Granger over transfer entropy*
7. **Addison, P.S.** (2002). *The Illustrated Wavelet Transform Handbook: Introductory Theory and Applications in Science, Engineering, Medicine and Finance*. IOP Publishing. — *Finance-specific wavelet applications*

---

*Document prepared April 9, 2026. Good luck, Josh.*
