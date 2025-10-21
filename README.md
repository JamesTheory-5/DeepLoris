# DeepLoris

# Differentiable Loris (DLoris): the DDSP-next architecture

Here’s a concrete, buildable plan to fuse **Loris** (phase-correct analysis, reassignment, partial tracking) with **PyTorch DDSP** (end-to-end differentiability & learning).

---

## 1) System at a glance

```
x(t) ──► (A) Analysis Frontend ──► Z (partials+noise params) ──► (B) Neural Mapper ──► Z'
                                   ▲                                     │
                                   │                                     ▼
                            (C) Differentiable Synth  ◄────────────── Losses(x, x̂)
```

* **(A) Analysis Frontend** (Loris-style): produces *teacher* parameters
  `Z = {t_i, f_i(t), a_i(t), φ_i(t), B_i(t) } + noise band`
* **(B) Neural Mapper**: learns semantics; predicts `Z'` from conditioning (text, MIDI, latent) or from x via encoder
* **(C) Differentiable Synth**: PyTorch/JAX additive+noise, **phase-coherent**; enables backprop

Two training modes:

1. **Teacher-forced** (supervised): train networks to match Loris parameters (`Z' ≈ Z`)
2. **Perceptual reconstruction** (self-supervised): synthesize `x̂` from `Z'` and minimize spectral/phase losses vs `x`

---

## 2) Analysis frontend (non-differentiable, offline)

Use your current code (reassignment → consensus → peak picking → tracking → phase correction).

Outputs per clip:

* **Partials:** `t_i ∈ ℝ^T`, `f_i(t) ∈ ℝ^T`, `a_i(t) ∈ ℝ^T`, `φ_i(t) ∈ ℝ^T`, optional **bandwidth** `B_i(t)`
* **Residual noise:** multiband energies `n_b(t)` (e.g., ERB bands)

Preprocess for learning:

* Resample to a **frame rate** (e.g., 250 Hz) → tensors of shape `[P, T, F]` where `F` ~ {amplitude, frequency, bandwidth, (optional) phase increment}
* Normalize: `log a`, `Δφ = unwrap(φ)` differences, `z-score` per instrument
* Pad/pack to fixed `P_max` with masks (keep top-energy tracks)

> These become **teacher trajectories**. Store as `.npz` per clip.

---

## 3) Differentiable synthesizer (PyTorch)

Phase-coherent additive bank + filtered noise. Stable, fast, vectorized.

```python
import torch, torch.nn as nn, torch.nn.functional as F
PI = 3.141592653589793

class AdditiveSynth(nn.Module):
    def __init__(self, fs, n_partials, frame_rate):
        super().__init__()
        self.fs = fs
        self.h = int(fs // frame_rate)  # samples per frame

    def forward(self, f_hz, a_lin, phi0, mask):
        """
        f_hz, a_lin, phi0: [B, P, T]; mask: [B, P, T] (0/1)
        returns waveform: [B, N]
        """
        B,P,T = f_hz.shape
        # upsample per frame → per sample (linear or spline)
        N = T * self.h
        t = torch.linspace(0, T-1, N, device=f_hz.device)  # frame-time per sample
        def up(x):  # x: [B,P,T]
            return F.interpolate(x, size=N, mode='linear', align_corners=True)
        f_s   = up(f_hz)
        a_s   = up(a_lin)
        m_s   = up(mask)
        phi0s = up(phi0)

        # integrate frequency to phase (sample domain)
        dphi  = 2*PI * f_s / self.fs                          # [B,P,N]
        phi   = torch.cumsum(dphi, dim=-1) + phi0s[..., :1]   # anchor
        y_p   = a_s * torch.sin(phi) * m_s
        y     = y_p.sum(dim=1)                                # sum partials
        return y

class NoiseSynth(nn.Module):
    def __init__(self, fs, bands, frame_rate):
        super().__init__()
        self.fs, self.h = fs, int(fs // frame_rate)
        self.filt = nn.Conv1d(bands, 1, kernel_size=1)  # optional mixing

    def forward(self, band_envs):
        # band_envs: [B, BANDS, T] → upsample to N, excite with white noise
        B,BK,T = band_envs.shape
        N = T * self.h
        env = F.interpolate(band_envs, size=N, mode='linear', align_corners=True)
        noise = torch.randn(B, BK, N, device=env.device) / (BK**0.5)
        y = self.filt(env * noise).squeeze(1)
        return y

class DLorisSynth(nn.Module):
    def __init__(self, fs, n_partials, bands, frame_rate=250):
        super().__init__()
        self.add = AdditiveSynth(fs, n_partials, frame_rate)
        self.noi = NoiseSynth(fs, bands, frame_rate)

    def forward(self, params):
        y_add = self.add(params['f_hz'], params['a_lin'], params['phi0'], params['mask'])
        y_noi = self.noi(params['band_envs'])
        return (y_add + y_noi) * 0.5
```

**Notes**

* `phi0` can be learned or taken from teacher `φ` (initialize, then let network predict offsets)
* Optional **bandwidth** → amplitude-dependent damping or sinusoid band-limited kernels
* Use **chunked rendering** for long clips (no OOM)

---

## 4) Neural mapper (encoders/decoders)

### A) Supervised (teacher-forced)

* **Input:** teacher trajectories (optionally + conditioning: MIDI, text, score)
* **Target:** same trajectories
* **Model:** lightweight Transformer over time (per partial) with cross-attention over conditioning context

Tensor layout:

* `X_in = [log a, f, Δf, (B), mask]` → `[B, P, T, D]`
* **Architecture:**

  * Per-partial MLP → positional encoding → Transformer (shared weights across partials)
  * Head predicts `a'`, `f'`, `B'`, `phi0'`
  * Global latent (from text/MIDI) via cross-attn for semantics

### B) Self-supervised (reconstruction)

* **Encoder:** ConvNet on audio or spectrogram → latent `z`
* **Decoder:** predicts synth params `Z'` → **Synth** → waveform `x̂`
* **Losses:** multi-scale STFT + phase-aware + trajectory regularizers

---

## 5) Losses (balanced, stable)

* **Multi-resolution STFT mag:** `L_mag = Σ || |STFT(x)| - |STFT(x̂)| ||_1`
* **Phase gradient / reassignment consistency (lightweight):**
  Compute **instantaneous frequency** via analytic signal (Hilbert) at coarse scale and penalize `IF(x̂)` vs `IF_target` where stable.
* **Trajectory smoothness:**
  `L_tv = λ_f * TV(f'(t)) + λ_a * TV(log a'(t))`
* **Teacher parameter loss (if available):**
  `L_sup = ||f' - f||_1 + ||log a' - log a||_1 + ...` (masked)
* **Noise/energy budget:** encourage `y_add` + `y_noi` to match band energies

Total: `L = α L_mag + β L_phase + γ L_tv + δ L_sup + ε L_band`

---

## 6) Making the analysis “differentiable enough”

You have three practical paths:

1. **Teacher forcing only:** don’t backprop through analysis. Fastest to production.
2. **Straight-Through Estimator (STE):** use Loris params in forward; pass encoder gradients straight-through. Simple, often works.
3. **Learned surrogate of analysis:** train a network to predict Loris params from audio, supervised by your offline analyzer. Later, optionally distill to make it tighter.

Start with **(1)**; add **(3)** once the synth & losses are stable.

---

## 7) Real-time & deployment

* **Frame rate:** 200–400 Hz gives good control; hop = fs//frame_rate
* **Latency:** 1–2 frames (5–10 ms) with streaming additive render
* **Chunked synthesis:** overlap 2–4 frames with crossfade to kill seams
* **Optimization:**

  * TorchScript / `torch.compile`
  * Use fused kernels for upsampling + sin (or approximate `sin` with a Chebyshev polynomial if you must)
  * Optional CUDA FFT for residual spectral shaping

---

## 8) Data & batching

* Pack by **(B, P_max, T_max, D)** with masks
* Curriculum: start with **monophonic, quasi-harmonic** instruments (flute/violin) → add polyphonic textures
* Auto-augment: pitch scale, time-stretch, SNR mixes (apply to audio & warp teacher params accordingly)

---

## 9) Evaluation

* **Objective:** multi-res spectral distance, **IF RMSE** on stable partials, **F0** accuracy when applicable
* **Subjective:** MUSHRA/ABX vs. DDSP baselines, artifacts (phasiness, pre-echo), timbral plausibility
* **Control tests:** exact pitch-shift, time-stretch, morphing consistency (parameters manipulated → expected audio response)

---

## 10) Minimal training loop (supervised)

```python
synth = DLorisSynth(fs=48000, n_partials=Pmax, bands=32, frame_rate=250).to(dev)
net   = MapperModel(...).to(dev)  # your Transformer/MLP stack
opt   = torch.optim.AdamW(list(net.parameters())+list(synth.parameters()), lr=3e-4)

for batch in loader:  # batch has audio x, teacher params Z, masks
    x = batch['audio'].to(dev)              # [B,N]
    Z = {k: v.to(dev) for k,v in batch['teacher'].items()}
    Z_pred = net(batch['cond'])             # predict params
    # mix teacher & pred early in training (scheduled sampling)
    Z_used = blend(Z_pred, Z, p_teacher=0.7)
    x_hat  = synth(Z_used)

    loss = stft_loss(x, x_hat) + traj_tv_loss(Z_used, batch['mask']) + sup_loss(Z_pred, Z, batch['mask']) * 0.5
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step()
```

---

## 11) Roadmap (pragmatic)

1. **Phase 1 (2–3 weeks):** implement synth, train on teacher params → perfect reconstruction
2. **Phase 2:** supervised predictor from spectrogram → params (no audio loss yet)
3. **Phase 3:** add waveform/perceptual loss via synth; tune multi-res STFT
4. **Phase 4:** add conditioning (MIDI/text), real-time stream, export

---

## 12) What you gain over DDSP

* **True phase coherence** (no “phasiness” in complex spectra)
* **Accurate IF & onset localization** via reassignment heritage
* **High partial counts** beyond harmonic stacks (inharmonic/metallic content)
* **Semantic control** on **physically meaningful** parameters
* A path to **interpretable generative audio** (transformers over partials)

---

If you want, I can flesh out the **MapperModel** (a compact per-partial Transformer with masking, cross-attn to MIDI/text) and a **ready-to-run multi-res STFT loss** snippet next.
