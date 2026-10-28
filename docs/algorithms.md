# FinSignal Suite - Mathematical Algorithms Handbook

This document outlines the core algorithms implemented in the FinSignal Suite DSP engine. It is provided for hackathon judges to verify the rigorous mathematics underlying our financial models.

## 1. Multi-Resolution Analysis (MRA) via Maximum Overlap DWT
In traditional finance, asset price series $P(t)$ are often assumed to be non-stationary. We transform prices to log-returns $r(t)=\ln(P(t)/P(t-1))$, but variance is still non-constant cross-scale.
We employ the **Maximum Overlap Discrete Wavelet Transform (MODWT)** because, unlike the standard DWT, MODWT is shift-invariant and preserves the length of the time series at all scales, which is critical for pointwise time-alignment with financial backtesting.

Given the signal $X_t$, the scale $j$ detail $D_{j,t}$ and approximation $S_{j,t}$ are computed via filtering with scale-dependent wavelet filters $\tilde{h}_{j,l}$ and scaling filters $\tilde{g}_{j,l}$:

$$D_{j,t}=\sum_{l=0}^{L_j-1}\tilde{h}_{j,l}X_{t-l\bmod N}$$
$$S_{j,t}=\sum_{l=0}^{L_j-1}\tilde{g}_{j,l}X_{t-l\bmod N}$$

We dynamically pad the array boundary by mirroring (`reflect`) to prevent boundary discontinuities and adapt to Python `pywt` exact length modulus constraints without dropping data points. 

## 2. Wavelet Coherence (Cross-Spectrum)
To measure lead-lag and correlation dynamics between two log-return series $X$ and $Y$ at specific frequencies, we use the continuous wavelet transform cross-spectrum.
Let $W_n^X(s)$ and $W_n^Y(s)$ be the CWTs of the two series at scale $s$. The Cross Wavelet Spectrum is $W_n^{XY}(s)=W_n^X(s)W_n^{Y*}(s)$.

We calculate the localized squared **Wavelet Coherence**:

$$R_n^2(s)=\frac{\left|S(s^{-1}W_n^{XY}(s))\right|^2}{S(s^{-1}|W_n^X(s)|^2)\cdot S(s^{-1}|W_n^Y(s)|^2)}$$

where $S$ is a smoothing operator in both time and scale. Values near 1 specify high shared local variance. Phase angles $\theta=\arctan(\Im(W^{XY}),\Re(W^{XY}))$ dictate lead/lag structures.

## 3. Spectral Granger Causality (Geweke 1982)
When the phase coherence indicates a lead, we mathematically formalize causality using Spectral Granger Causality. For a bivariate VAR($p$) process:

$$\begin{bmatrix}X_t\\Y_t\end{bmatrix}=\sum_{k=1}^p\Theta_k\begin{bmatrix}X_{t-k}\\Y_{t-k}\end{bmatrix}+\begin{bmatrix}\epsilon_{xt}\\\epsilon_{yt}\end{bmatrix}$$

The spectral density matrix is generated via the Fourier transform of the moving average representation $S(\omega)=H(\omega)\Sigma H^*(\omega)$.
Geweke's measure of causality from $Y\to X$ at frequency $\omega$ is defined as:

$$f_{Y\to X}(\omega)=\ln\left[\frac{S_{xx}(\omega)}{S_{xx}(\omega)-[H_{xy}(\omega)]^2\Sigma_{yy|x}}\right]$$

This indicates the proportion of the cross-spectrum driving $X$ strictly originating from $Y$'s historical shocks.
