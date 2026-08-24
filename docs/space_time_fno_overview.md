# Space-Time Factorized FNO: Presentation Overview

A clean, presentation-ready overview of the **Space-Time Factorized Fourier Neural Operator** (`FNOInterpolate`). This document summarizes the end-to-end forecasting pipeline and core operator mechanics into two high-level, easily digestible diagrams suitable for meetings and slide decks.

---

## 1. End-to-End Prediction Pipeline

The model transforms unstructured mesh inputs into continuous space-time forecasts through a three-stage **Encode $\to$ Process $\to$ Decode** pipeline.

```mermaid
flowchart LR
    subgraph INPUT ["1. Irregular Mesh Input"]
        direction TB
        in_pts["<b>Unstructured 3D Point Cloud</b><br/>• History: $T_{\text{in}} = 10$ steps<br/>• Features: Concentrations, Heads, Pumping"]
    end

    subgraph ENCODE ["2. Grid Encoding"]
        direction TB
        enc["<b>Nearest-Neighbor Mapping</b><br/>Projects irregular nodes onto<br/>regular 4D Latent Grid<br/><code>(24 × 24 × 12 × 10)</code>"]
    end

    subgraph BACKBONE ["3. Space-Time FNO"]
        direction TB
        fno["<b>Factorized Spectral Backbone</b><br/>• 2-Layer Lifting MLP<br/>• 4× Space-Time Conv Blocks<br/>• Decoupled 3D Space + 1D Time"]
    end

    subgraph DECODE ["4. Continuous Decoding"]
        direction TB
        dec["<b>3D Trilinear Sampling</b><br/><code>F.grid_sample()</code> decodes<br/>latent grid back to arbitrary<br/>query coordinates"]
    end

    subgraph OUTPUT ["5. Output & Loss"]
        direction TB
        out["<b>Forecast & Training Loss</b><br/>• Future Horizon: $T_{\text{out}} = 10$<br/>• Projection MLP (Conc, Head)<br/>• Variance-Aware Core Loss"]
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
    in_act["<b>Input Latent Tensor</b> $H^{(l-1)}$<br/>Shape: <code>(Batch, Channels=64, X=24, Y=24, Z=12, T=10)</code>"]

    subgraph FACTORIZED_CONV ["Factorized Space-Time Spectral Operator"]
        direction LR
        
        subgraph SPACE ["Path A: 3D Spatial Spectral Conv"]
            direction TB
            s_fft["3D Real FFT $(X, Y, Z)$"] --> s_mode["Truncate Spatial Modes<br/>$(k_x=10, k_y=10, k_z=6)$"] --> s_weight["Complex Low-Rank Weights<br/>(TensorLy-Torch)"] --> s_ifft["3D Inverse Real FFT"]
        end

        subgraph TIME ["Path B: 1D Temporal Spectral Conv"]
            direction TB
            t_fft["1D Real FFT $(T)$"] --> t_mode["Truncate Temporal Modes<br/>$(k_t=6)$"] --> t_weight["Complex Low-Rank Weights<br/>(TensorLy-Torch)"] --> t_ifft["1D Inverse Real FFT"]
        end

        subgraph SKIP ["Path C: Linear Skip"]
            direction TB
            skip_conv["Pointwise Conv3D<br/><code>(1×1×1 Kernel)</code>"]
        end
    end

    in_act --> SPACE
    in_act --> TIME
    in_act --> SKIP

    SPACE & TIME --> add_spectral["<b>Spectral Addition</b><br/>$\mathcal{K}_{\text{space}} + \mathcal{K}_{\text{time}}$"]
    add_spectral & SKIP --> add_all["<b>Residual Sum</b><br/>$\mathcal{K}_{\text{ST}} + W_{\text{skip}}$"]
    add_all --> act["<b>Activation</b><br/>GELU"] --> out_act["<b>Output Latent Tensor</b> $H^{(l)}$"]

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

## 3. High-Level Specification Summary

| Component | Choice | Purpose |
| :--- | :--- | :--- |
| **Grid Resolution** | $(N_x, N_y, N_z, N_t) = (24, 24, 12, 10)$ | Regular 4D latent lattice for FFT operations |
| **Spectral Modes** | Spatial: $(10, 10, 6)$ \| Temporal: $6$ | Captures dominant low-to-mid frequency dynamics |
| **Hidden Channels** | $d_{\text{hidden}} = 64$ | Latent representation dimension across 4 FNO blocks |
| **Decoding Method** | 3D Trilinear `F.grid_sample` | Continuous, differentiable query interpolation |
| **Predicted Fields** | `mass_concentration`, `head` | Multi-column simultaneous subsurface simulation |
| **Loss Function** | Relative $L_2$ + Variance-Aware Concentration | Focuses gradient signals on high-dynamic plume fronts |
