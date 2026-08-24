# Space-Time Factorized FNO: Presentation Overview

A clean, presentation-ready overview of the **Space-Time Factorized Fourier Neural Operator** (`FNOInterpolate`). This document summarizes the end-to-end forecasting pipeline, core operator mechanics, and loss formulation into high-level, easily digestible diagrams and sections suitable for meetings and slide decks.

---

## 1. End-to-End Prediction Pipeline

The model transforms unstructured mesh inputs into continuous space-time forecasts through a three-stage **Encode $\to$ Process $\to$ Decode** pipeline.

```mermaid
flowchart LR
    subgraph INPUT ["1. Irregular Mesh Input"]
        direction TB
        in_pts["<b>Unstructured 3D Point Cloud</b><br/>• History: <i>T</i><sub>in</sub> = 10 steps<br/>• Features: <i>X</i> ∈ ℝ<sup>(<i>B</i> × <i>N</i><sub>pts</sub>·<i>T</i><sub>in</sub> × 10)</sup><br/>• Conc, Head, Pumping Rates"]
    end

    subgraph ENCODE ["2. Grid Encoding"]
        direction TB
        enc["<b>Nearest-Neighbor Mapping</b><br/>Projects irregular nodes onto<br/>regular 4D Latent Grid<br/><code>(24 × 24 × 12 × 10)</code>"]
    end

    subgraph BACKBONE ["3. Space-Time FNO"]
        direction TB
        fno["<b>Factorized Spectral Backbone</b><br/>• 2-Layer Lifting MLP (10 → 64)<br/>• 4× Factorized 𝒦<sub>ST</sub> Conv Blocks<br/>• Decoupled 3D Space + 1D Time"]
    end

    subgraph DECODE ["4. Continuous Decoding"]
        direction TB
        dec["<b>3D Trilinear Sampling</b><br/><code>F.grid_sample()</code> decodes<br/>latent grid back to continuous<br/>physical query coordinates"]
    end

    subgraph OUTPUT ["5. Output & Loss"]
        direction TB
        out["<b>Forecast & Core Loss</b><br/>• Horizon: <i>T</i><sub>out</sub> = 10 steps<br/>• Predictions: <i>Ŷ</i> ∈ ℝ<sup>(<i>B</i> × <i>N</i> × 2)</sup><br/>• ℒ = (1−λ)ℒ<sub>global</sub> + λℒ<sub>conc-var</sub>"]
    end

    INPUT --> ENCODE --> BACKBONE --> DECODE --> OUTPUT

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:1.5px;
    classDef highlight fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    classDef accent fill:#fdf4ff,stroke:#c026d3,stroke-width:2px;
    classDef success fill:#ecfdf5,stroke:#059669,stroke-width:2px;

    class INPUT default;
    class ENCODE,DECODE highlight;
    class BACKBONE accent;
    class OUTPUT success;
```

### Key Takeaways
1. **Mesh-Agnostic Processing**: Encodes irregular point clouds into a regular 4D lattice, runs efficient Fourier convolutions on the grid, and samples back to arbitrary physical coordinates.
2. **Pushforward Multi-Step Rollout**: Ingests 10 historical timesteps with future pumping forcings and forecasts the next 10 consecutive timesteps in a single forward pass.
3. **Core vs. Ghost Node Loss**: Evaluates across the entire patch for boundary-smooth interpolation, while restricting loss optimization to internal core nodes to eliminate boundary artifacts.

---

## 2. Space-Time Factorized Spectral Conv Block

Inside each backbone block, expensive 4D spectral convolution is factorized into parallel 3D spatial and 1D temporal Fourier convolutions, drastically reducing computational complexity while retaining global space-time receptive fields.

```mermaid
flowchart TD
    in_act["<b>Input Latent Tensor</b> <i>H</i><sup>(<i>l</i>−1)</sup><br/>Shape: <code>(Batch, Channels=64, X=24, Y=24, Z=12, T=10)</code>"]

    subgraph FACTORIZED_CONV ["Factorized Space-Time Spectral Operator"]
        direction LR
        
        subgraph SPACE ["Path A: 3D Spatial Spectral Conv"]
            direction TB
            s_fft["3D Real FFT ℱ<sub>3D</sub>(X, Y, Z)"] --> s_mode["Truncate Spatial Modes<br/>(<i>k<sub>x</sub></i>=10, <i>k<sub>y</sub></i>=10, <i>k<sub>z</sub></i>=6)"] --> s_weight["Complex Weights <i>R</i><sub>space</sub><br/>(Low-Rank Factorized)"] --> s_ifft["3D Inverse Real FFT ℱ<sub>3D</sub><sup>−1</sup>"]
        end

        subgraph TIME ["Path B: 1D Temporal Spectral Conv"]
            direction TB
            t_fft["1D Real FFT ℱ<sub>1D</sub>(T)"] --> t_mode["Truncate Temporal Modes<br/>(<i>k<sub>t</sub></i>=6)"] --> t_weight["Complex Weights <i>R</i><sub>time</sub><br/>(Low-Rank Factorized)"] --> t_ifft["1D Inverse Real FFT ℱ<sub>1D</sub><sup>−1</sup>"]
        end

        subgraph SKIP ["Path C: Linear Skip"]
            direction TB
            skip_conv["Pointwise Conv3D <i>W</i><sub>skip</sub><br/><code>(1×1×1 Kernel)</code>"]
        end
    end

    in_act --> SPACE
    in_act --> TIME
    in_act --> SKIP

    SPACE & TIME --> add_spectral["<b>Spectral Addition</b><br/>𝒦<sub>space</sub> + 𝒦<sub>time</sub>"]
    add_spectral & SKIP --> add_all["<b>Residual Sum</b><br/>𝒦<sub>ST</sub> + <i>W</i><sub>skip</sub>"]
    add_all --> act["<b>Activation</b><br/>GELU"] --> out_act["<b>Output Latent Tensor</b> <i>H</i><sup>(<i>l</i>)</sup>"]

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:1.5px;
    classDef spaceStyle fill:#eff6ff,stroke:#3b82f6,stroke-width:1.5px;
    classDef timeStyle fill:#fdf4ff,stroke:#c026d3,stroke-width:1.5px;
    classDef skipStyle fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px;
    classDef sumStyle fill:#fffbeb,stroke:#d97706,stroke-width:2px;

    class in_act,out_act,act default;
    class SPACE spaceStyle;
    class TIME timeStyle;
    class SKIP skipStyle;
    class add_spectral,add_all sumStyle;
```

### Architectural Benefits
- **Linear Complexity Scaling in Time**: Splitting 4D FFT into $(3\text{D} + 1\text{D})$ avoids the cubic-to-quartic mode expansion of monolithic 4D Fourier layers.
- **Low-Rank Parameter Factorization**: Uses `tltorch` tensor factorization (rank = 0.5) to keep model parameter size compact while preventing overfitting.
- **Non-Local Space-Time Coupling**: Each block simultaneously captures global spatial dispersion patterns and long-term temporal trends in the frequency domain.

---

## 3. Variance-Aware Multi-Column Loss Computation

To ensure stable multi-step rollouts and prevent gradient vanishing around critical contamination zones, training uses a compound objective with **domain decomposition masking**, **pushforward temporal weighting**, and **variance-aware focus**.

```mermaid
flowchart LR
    preds["<b>Raw Prediction <i>Ŷ</i></b><br/><code>(B, N_pts, T_out, 2)</code>"] --> core_slice["<b>1. Core Domain Slicing</b><br/>Extract <code>Ŷ[:, :N_core, :, :]</code><br/>(Exclude buffer ghost nodes)"]
    
    core_slice --> global_loss["<b>2. Global Relative L₂ Loss</b><br/>Measures bulk error across<br/>both Conc & Head"]
    core_slice --> var_loss["<b>3. Variance-Aware Conc Loss</b><br/>Weights errors by spatial<br/>variance <b>w</b><sub>spatial</sub> (Plume front)"]
    
    time_w["<b>Pushforward Time Weights</b><br/><i>w<sub>t</sub></i> ∈ [1.0, 2.0] across <i>T</i><sub>out</sub>"] -.-> global_loss & var_loss

    global_loss & var_loss --> total_loss["<b>4. Total Combined Loss</b><br/>ℒ = (1 − λ)ℒ<sub>global</sub> + λℒ<sub>conc-var</sub>"]

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:1.5px;
    classDef sliceStyle fill:#fdf4ff,stroke:#c026d3,stroke-width:1.5px;
    classDef lossStyle fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px;
    classDef finalStyle fill:#ecfdf5,stroke:#059669,stroke-width:2px;

    class preds,time_w default;
    class core_slice sliceStyle;
    class global_loss,var_loss lossStyle;
    class total_loss finalStyle;
```

### Mathematical Formulation

1. **Core Domain Masking**:
   
   $$
   \hat{\mathbf{Y}}_{\text{core}} = \hat{\mathbf{Y}}[:, :N_{\text{core}}, :, :]
   $$
   
   Eliminates artificial edge effects created by patch boundaries during domain decomposition.

2. **Pushforward Temporal Weighting**:
   
   $$
   w_t = 1.0 + \frac{t - 1}{T_{\text{out}} - 1}, \quad t \in \{1, \dots, T_{\text{out}}\}
   $$
   
   Linearly increases penalty from $1.0\times$ at $t=1$ to $2.0\times$ at $t=10$ to prevent compounding autoregressive rollout errors.

3. **Global Relative $L_2$ Loss**:
   
   $$
   \mathcal{L}_{\text{global}} = \frac{\sum_{t=1}^{T_{\text{out}}} w_t \cdot \frac{\Vert\hat{\mathbf{Y}}_t - \mathbf{Y}_t\Vert_2}{\Vert\mathbf{Y}_t\Vert_2 + \epsilon}}{\sum_{t=1}^{T_{\text{out}}} w_t}
   $$
   
   Provides balanced optimization across both physical variables (`mass_concentration` and `hydraulic_head`).

4. **Variance-Aware Concentration Loss**:
   
   $$
   \mathcal{L}_{\text{conc-var}} = \frac{\sum_{t=1}^{T_{\text{out}}} w_t \cdot \frac{\Vert(\hat{\mathbf{C}}_t - \mathbf{C}_t) \odot \sqrt{\mathbf{w}_{\text{spatial}}}\Vert_2}{\Vert\mathbf{C}_t \odot \sqrt{\mathbf{w}_{\text{spatial}}}\Vert_2 + \epsilon}}{\sum_{t=1}^{T_{\text{out}}} w_t}
   $$
   
   $\mathbf{w}_{\text{spatial}}$ are pre-computed normalized temporal variances that focus gradient energy on fast-moving, high-gradient contaminant plume fronts rather than static background regions.

5. **Total Combined Loss**:
   
   $$
   \mathcal{L}_{\text{total}} = (1 - \lambda) \mathcal{L}_{\text{global}} + \lambda \mathcal{L}_{\text{conc-var}} \quad (\lambda = 0.5)
   $$

---

## 4. High-Level Specification Summary

| Component | Choice | Purpose |
| :--- | :--- | :--- |
| **Grid Resolution** | $(N_x, N_y, N_z, N_t) = (24, 24, 12, 10)$ | Regular 4D latent lattice for FFT operations |
| **Spectral Modes** | Spatial: $(10, 10, 6)$, Temporal: $6$ | Captures dominant low-to-mid frequency dynamics |
| **Hidden Channels** | $d_{\text{hidden}} = 64$ | Latent representation dimension across 4 FNO blocks |
| **Decoding Method** | 3D Trilinear `F.grid_sample` | Continuous, differentiable query interpolation |
| **Predicted Fields** | `mass_concentration`, `head` | Multi-column simultaneous subsurface simulation |
| **Pushforward Horizon**| $T_{\text{in}} = 10 \to T_{\text{out}} = 10$ | Multi-step rollout in a single forward pass |
| **Loss Function** | Relative $L_2$ + Variance-Aware Concentration | Penalizes long-term drift and sharp plume front errors |
