# T-PATCHGNN Architecture Summary

**Paper**: Zhang et al., "Irregular Multivariate Time Series Forecasting:
A Transformable Patching Graph Neural Networks Approach", ICML 2024.

## Architecture (adapted for c5_Q-state)

### Input
- 5 jointly-modeled variables: QS_1..QS_5 (order statistics from draws on {1..39})
- Lag window: 10 observations per variable
- Classification over 39 discrete classes (values 1..39) per variable

### Component 1: Transformable Patching
- Patch size: 5 observations (fixed temporal span)
- Number of patches: 2
- Each variable patched independently (avoids pre-alignment explosion)

### Component 2: Patch Encoding
- **Continuous time embedding** (Eq. 3): `phi(t)[d] = omega_0*t + alpha_0` (d=0), `sin(omega_d*t + alpha_d)` (d>0)
- Embedding dimension: D_t = 10
- **TTCN** (Transformable Time-aware Convolution Network):
  - Meta-filter generates conv filters dynamically conditioned on input (Eq. 5)
  - Softmax normalization along temporal dimension
  - Produces patch embedding h_p = [h_c || m_p] (Eq. 7)

### Component 3: Intra-Time Series (Transformer)
- Multi-head self-attention with H=1 heads (Eq. 8)
- Positional encoding added to patch tokens
- LayerNorm + FFN residual blocks

### Component 4: Inter-Time Series (Time-Adaptive GNN)
- **Time-varying adaptive graph learning** (Eq. 9-10):
  - Static variable embeddings E_s (D_g = 10)
  - Dynamic patch embeddings via gated adding
  - A_p = Softmax(ReLU(E_p1 * E_p2^T))
- **GNN message passing** (Eq. 11):
  - H_p = ReLU(sum_m A_p^m * H_tf * W_gnn_m)
  - M = 1 layer(s)

### Component 5: Forecasting Head
- Flatten patch representations (Eq. 12)
- Per-variable MLP: [H_n || phi(q)] -> 39-class logits (Eq. 13)
- Cross-entropy loss (adapted from MSE in paper for classification)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Hidden dimension | 64 |
| Time embedding dim | 10 |
| Graph embedding dim | 10 |
| Transformer heads | 1 |
| GNN layers | 1 |
| Stacked blocks | 1 |
| Patch size | 5 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Early stopping | patience 10 |

### Total Parameters
- Total: 126,001
- Trainable: 126,001
