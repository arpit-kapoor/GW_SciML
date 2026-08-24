# Space-Time Factorized FNO: Presentation Overview

A clean, presentation-ready overview of the **Space-Time Factorized Fourier Neural Operator** (`FNOInterpolate`). This document summarizes the end-to-end forecasting pipeline, core operator mechanics, and loss formulation into high-level, easily digestible diagrams and sections suitable for meetings and slide decks.

---

## 1. End-to-End Prediction Pipeline

The model transforms unstructured mesh inputs into continuous space-time forecasts through a three-stage **Encode $\to$ Process $\to$ Decode** pipeline.

```mermaid
flowchart LR
    subgraph INPUT ["1. Irregular Mesh Input"]
        direction TB
        in_pts["<b>3D Unstructured Cloud</b><br/>• History: T_in = 10 steps<br/>• Coords + Features (C_in=10)<br/>• Conc, Head, Pumping"]
    end

    subgraph ENCODE ["2. Grid Encoding"]
        direction TB
        enc["<b>Nearest-Neighbor Mapping</b><br/>Interpolates points to<br/>4D Latent Regular Grid<br/>(24 × 24 × 12 × 10)"]
    end

    subgraph BACKBONE ["3. Space-Time FNO"]
        direction TB
        fno["<b>Factorized Spectral Backbone</b><br/>• 2-Layer Lifting (10 → 64)<br/>• 4× Space-Time Conv Blocks<br/>• Decoupled 3D Space + 1D Time"]
    end

    subgraph DECODE ["4. Continuous Decoding"]
        direction TB
        dec["<b>3D Trilinear Sampling</b><br/>F.grid_sample() queries to<br/>continuous physical points<br/>across horizon T_out = 10"]
    end

    subgraph OUTPUT ["5. Output & Loss"]
        direction TB
        out["<b>Forecast & Core Loss</b><br/>• 2-Layer Projection (64 → 2)<br/>• Outputs: Conc, Head<br/>• Variance-Aware Core Loss"]
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
    in_act["<b>Input Latent Activation H⁽ˡ⁻¹⁾</b><br/>Channels: 64<br/>Grid: (Nx=24, Ny=24, Nz=12, Nt=10)"]

    subgraph FACTORIZED_CONV ["Factorized Space-Time Spectral Operator"]
        direction LR
        
        subgraph SPACE ["Path A: 3D Spatial Spectral Conv"]
            direction TB
            s_fft["<b>3D Real FFT</b><br/>rfftn(dim=[-3, -2, -1])"] --> s_mode["<b>Truncate Modes</b><br/>(kx=10, ky=10, kz=6)"] --> s_weight["<b>Low-Rank Weights</b><br/>R_space (TensorLy-Torch)"] --> s_ifft["<b>3D Inverse FFT</b><br/>irfftn(dim=[-3, -2, -1])"]
        end

        subgraph TIME ["Path B: 1D Temporal Spectral Conv"]
            direction TB
            t_fft["<b>1D Real FFT</b><br/>rfftn(dim=[-1])"] --> t_mode["<b>Truncate Modes</b><br/>(kt=6)"] --> t_weight["<b>Low-Rank Weights</b><br/>R_time (TensorLy-Torch)"] --> t_ifft["<b>1D Inverse FFT</b><br/>irfftn(dim=[-1])"]
        end

        subgraph SKIP ["Path C: Linear Skip"]
            direction TB
            skip_conv["<b>Pointwise Conv3D</b><br/>W_skip (1×1×1 Kernel)"]
        end
    end

    in_act --> SPACE
    in_act --> TIME
    in_act --> SKIP

    SPACE & TIME --> add_spectral["<b>Spectral Addition</b><br/>K_space + K_time"]
    add_spectral & SKIP --> add_all["<b>Residual Sum</b><br/>K_ST + W_skip"]
    add_all --> act["<b>Activation</b><br/>GELU"] --> out_act["<b>Output Latent Activation H⁽ˡ⁾</b><br/>Channels: 64<br/>Grid: (Nx=24, Ny=24, Nz=12, Nt=10)"]

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
