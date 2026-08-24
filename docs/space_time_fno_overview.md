# Space-Time Factorized FNO: Presentation Overview

A clean, presentation-ready overview of the **Space-Time Factorized Fourier Neural Operator** (`FNOInterpolate`). This document summarizes the end-to-end forecasting pipeline, core operator mechanics, and loss formulation into high-level, easily digestible diagrams and sections suitable for meetings and slide decks.

---

## 1. End-to-End Prediction Pipeline

The model transforms unstructured mesh inputs into continuous space-time forecasts through a three-stage **Encode $\to$ Process $\to$ Decode** pipeline.

```mermaid
flowchart TD
    subgraph S1 ["Stage 1: Irregular Mesh Input"]
        in_pts["Unstructured 3D Point Cloud<br/>History: Tin = 10 timesteps<br/>Features: Conc, Head, Pumping"]
    end

    subgraph S2 ["Stage 2: 4D Grid Encoding"]
        enc["Nearest-Neighbor Assignment<br/>Interpolates nodes to regular grid<br/>Grid Shape: (24 × 24 × 12 × 10)"]
    end

    subgraph S3 ["Stage 3: Space-Time FNO Backbone"]
        fno["Lifting MLP (10 to 64 channels)<br/>4x Factorized FNO Conv Blocks<br/>Decoupled (3D Space + 1D Time)"]
    end

    subgraph S4 ["Stage 4: Continuous Query Decoding"]
        dec["3D Trilinear grid_sample()<br/>Decodes latent grid to query points<br/>Forecast Horizon: Tout = 10"]
    end

    subgraph S5 ["Stage 5: Output Projection & Loss"]
        out["Projection MLP (64 to 2 channels)<br/>Predictions: Conc and Head<br/>Variance-Aware Loss on Core Nodes"]
    end

    S1 --> S2 --> S3 --> S4 --> S5

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:1.5px;
    classDef highlight fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    classDef accent fill:#fdf4ff,stroke:#c026d3,stroke-width:2px;
    classDef success fill:#ecfdf5,stroke:#059669,stroke-width:2px;

    class S1 default;
    class S2,S4 highlight;
    class S3 accent;
    class S5 success;
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
    in_act["Input Activation H(l-1)<br/>Shape: (B, 64, 24, 24, 12, 10)"]

    subgraph FACTORIZED_CONV ["Factorized Space-Time Spectral Operator"]
        direction LR
        
        subgraph SPACE ["Branch A: 3D Space Conv"]
            direction TB
            s_fft["3D Real FFT<br/>dim = [-3, -2, -1]"]
            s_mode["Truncate Modes<br/>(kx=10, ky=10, kz=6)"]
            s_weight["Low-Rank Weights<br/>R_space (tltorch)"]
            s_ifft["3D Inverse Real FFT<br/>dim = [-3, -2, -1]"]
            
            s_fft --> s_mode --> s_weight --> s_ifft
        end

        subgraph TIME ["Branch B: 1D Time Conv"]
            direction TB
            t_fft["1D Real FFT<br/>dim = [-1]"]
            t_mode["Truncate Modes<br/>(kt = 6)"]
            t_weight["Low-Rank Weights<br/>R_time (tltorch)"]
            t_ifft["1D Inverse Real FFT<br/>dim = [-1]"]
            
            t_fft --> t_mode --> t_weight --> t_ifft
        end

        subgraph SKIP ["Branch C: Skip Connection"]
            direction TB
            skip_conv["Pointwise Conv3D<br/>W_skip (1x1x1 kernel)"]
        end
    end

    in_act --> SPACE
    in_act --> TIME
    in_act --> SKIP

    SPACE & TIME --> add_spectral["Spectral Sum<br/>K_space + K_time"]
    add_spectral & SKIP --> add_all["Residual Sum<br/>K_ST + W_skip"]
    add_all --> act["Activation<br/>GELU"]
    act --> out_act["Output Activation H(l)<br/>Shape: (B, 64, 24, 24, 12, 10)"]

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

## 3. Loss Formulation

To ensure robust multi-step rollout without compounding drift or boundary distortion, the model is trained with a compound objective combining **interior domain masking**, **temporal pushforward weighting**, and **variance-aware plume focus**:

$$
\mathcal{L}_{\text{total}} = (1 - \lambda) \mathcal{L}_{\text{global}} + \lambda \mathcal{L}_{\text{conc-var}} \quad (\lambda = 0.5)
$$

During domain decomposition, patches contain ghost buffer nodes for smooth interpolation; however, the loss is computed strictly on interior core nodes $\hat{\mathbf{Y}}_{\text{core}} = \hat{\mathbf{Y}}[:, :N_{\text{core}}, :, :]$ to eliminate artificial edge artifacts. Across the forecast horizon $T_{\text{out}} = 10$, timesteps are weighted by a linear pushforward schedule $w_t = 1.0 + \frac{t - 1}{T_{\text{out}} - 1} \in [1.0, 2.0]$ that penalizes later time errors more heavily.

The **Global Relative $L_2$ Loss** provides balanced relative error optimization across both physical variables (`mass_concentration` and `hydraulic_head`):

$$
\mathcal{L}_{\text{global}} = \frac{\sum_{t=1}^{T_{\text{out}}} w_t \cdot \frac{\Vert\hat{\mathbf{Y}}_t - \mathbf{Y}_t\Vert_2}{\Vert\mathbf{Y}_t\Vert_2 + \epsilon}}{\sum_{t=1}^{T_{\text{out}}} w_t}
$$

The **Variance-Aware Concentration Loss** uses pre-computed normalized temporal variances $\mathbf{w}_{\text{spatial}}$ to amplify gradient energy on moving, high-gradient contaminant plume fronts:

$$
\mathcal{L}_{\text{conc-var}} = \frac{\sum_{t=1}^{T_{\text{out}}} w_t \cdot \frac{\Vert(\hat{\mathbf{C}}_t - \mathbf{C}_t) \odot \sqrt{\mathbf{w}_{\text{spatial}}}\Vert_2}{\Vert\mathbf{C}_t \odot \sqrt{\mathbf{w}_{\text{spatial}}}\Vert_2 + \epsilon}}{\sum_{t=1}^{T_{\text{out}}} w_t}
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
