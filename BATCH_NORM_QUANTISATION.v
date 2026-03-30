/**
 * ═════════════════════════════════════════════════════════════════════════════
 * BATCH NORMALIZATION FOLDING & QUANTISATION FOR FPGA
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * This document explains the offline pre-processing steps applied to the trained
 * CNN model before deployment on FPGA hardware.
 *
 * OVERVIEW
 * ────────
 * The trained TensorFlow/Keras model includes:
 *   1. Convolutional layers (Conv1D)
 *   2. Batch Normalization (BN) layers
 *   3. Activation functions (ReLU)
 *   4. Pooling layers (MaxPool, GlobalAveragePool)
 *   5. Dense classifier layer
 *
 * During INFERENCE (not training), Batch Normalization becomes a fixed linear
 * transform that can be mathematically fused into the preceding Conv layer.
 * This ELIMINATES the need for a separate BN module on the FPGA.
 *
 * Additionally, floating-point weights are quantized to INT8 (signed 8-bit) to
 * save storage and use efficient 8-bit multipliers on DSP blocks.
 *
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * PART I: BATCH NORMALIZATION FOLDING
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * MATHEMATICAL BACKGROUND
 * ───────────────────────
 *
 * Training time (with BN enabled):
 *   z = Conv(x)                    [floating-point accumulation]
 *   y = BN(z) = γ·(z − μ_batch)/σ_batch + β
 *     where μ_batch, σ_batch are computed from the current mini-batch
 *
 * Inference time (BN disabled → use running statistics):
 *   z = Conv(x)
 *   y = BN(z) = γ·(z − μ_running)/√(σ²_running + ε) + β
 *     = γ·(z − μ)/√(σ² + ε) + β
 *     where μ, σ are the running mean/variance from training
 *
 * This is a FIXED LINEAR TRANSFORM (no randomness).
 * We can rewrite it as:
 *   y = (γ/√(σ² + ε))·z − (γ/√(σ² + ε))·μ + β
 *   y = α·z + β_fused
 *
 *   where:
 *     α         = γ / √(σ² + ε)              [scale factor per filter]
 *     β_fused   = β − α·μ                    [fused bias per filter]
 *
 * ABSORPTION INTO CONV LAYER
 * ──────────────────────────
 *
 * A Conv1D layer without bias (use_bias=False) followed by BN can be fused as:
 *
 *   Original (unfused):
 *     z = Conv1D_weight * x          [shape: (kernel, in_ch, out_ch)]
 *     y = BN(z)
 *
 *   Fused:
 *     w_fused = Conv1D_weight * scale[np.newaxis, np.newaxis, :]
 *     b_fused = -μ * scale + β
 *     y = w_fused * x + b_fused      [same output, single operation]
 *
 * IMPLEMENTATION IN PYTHON (from the notebook)
 * ─────────────────────────────────────────────
 *
 *   def fold_bn_into_conv(conv_layer, bn_layer):
 *       W = conv_layer.get_weights()[0]           # (kernel, in_ch, out_ch)
 *       gamma = bn_layer.get_weights()[0]         # (out_ch,)
 *       beta  = bn_layer.get_weights()[1]         # (out_ch,)
 *       mu    = bn_layer.get_weights()[2]         # (out_ch,)   [running mean]
 *       var   = bn_layer.get_weights()[3]         # (out_ch,)   [running var]
 *       eps   = bn_layer.epsilon                  # typically 1e-3
 *
 *       std = √(var + eps)
 *       scale = gamma / std                       # (out_ch,)
 *       W_fused = W * scale[np.newaxis, np.newaxis, :]
 *       b_fused = beta - mu * scale
 *
 *       return W_fused, b_fused
 *
 * After this step:
 *   - All BN parameters (γ, β, μ, σ) are discarded
 *   - Conv weights are scaled
 *   - Conv bias absorbs the BN transformation
 *   - NO BN module needed on FPGA → saves hardware, reduces latency
 *
 * HARDWARE IMPLICATION
 * ───────────────────
 * On the FPGA:
 *   - Each Conv1D MAC (multiply-accumulate) produces scaled output automatically
 *   - Add the fused bias (pre-computed, constant)
 *   - Pass to ReLU or next layer
 *   - Zero additional cost compared to unfused BN
 *
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * PART II: QUANTISATION (Float32 → INT8)
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * PURPOSE
 * ───────
 * After training, all weights and biases are floating-point (float32).
 * For FPGA deployment, we quantize to INT8 (signed 8-bit, range −128…127).
 *
 * BENEFITS:
 *   1. Storage: 4× reduction (float32 = 4 bytes, int8 = 1 byte)
 *      → Fits more weights in on-chip BRAM
 *   2. Multipliers: 8-bit × 8-bit cheaper than 32-bit × 32-bit
 *      → DSP48 slices can process 4 INT8 MACs in parallel
 *   3. Bandwidth: 4× lower memory bandwidth requirement
 *   4. Accuracy: For shallow networks, quantisation error < 1% (usually)
 *
 * QUANTISATION FORMULA
 * ───────────────────
 *
 * Goal: Map float weight w ∈ [w_min, w_max] to int8 ∈ [−127, 127]
 *
 * Step 1 — Compute absolute maximum:
 *   w_absmax = max(|w_i|) for all weights in the tensor
 *
 * Step 2 — Compute scale factor:
 *   S = w_absmax / 127
 *   Interpretation: each INT8 unit represents S float32 units.
 *
 * Step 3 — Quantise:
 *   w_q = round(w / S)  clipped to [−127, 127]
 *
 * Step 4 — Verify (optional):
 *   w_approx = w_q * S   [reconstructed float value]
 *   error = |w − w_approx|
 *
 * EXAMPLE
 * ──────
 * Suppose Conv1 has weights in range [−0.5, +0.6] (typical for trained CNNs):
 *   w_absmax = 0.6
 *   S = 0.6 / 127 ≈ 0.00472
 *
 *   w[0] =  0.6  →  w_q[0] = round(0.6 / 0.00472) = 127     (max)
 *   w[1] = −0.5  →  w_q[1] = round(−0.5 / 0.00472) = −106
 *   w[2] =  0.3  →  w_q[2] = round(0.3 / 0.00472) =   64
 *
 *   Reconstructed:
 *   w[0]_approx = 127 × 0.00472 ≈ 0.5994   (error ≈ 0.0006)
 *   w[1]_approx = −106 × 0.00472 ≈ −0.5003  (error ≈ 0.0003)
 *   w[2]_approx = 64 × 0.00472 ≈ 0.3021    (error ≈ 0.0021)
 *
 * QUANTISATION ERROR DISTRIBUTION
 * ────────────────────────────────
 *
 * Maximum rounding error per weight: ±S/2
 *   For the example above: error_max = 0.00472 / 2 ≈ 0.00236
 *
 * For deep networks with many layers, these errors accumulate.
 * For our shallow 2-layer CNN (~740 parameters), quantisation typically
 * reduces inference accuracy by < 1–2%.
 *
 * IMPLEMENTATION IN PYTHON (from the notebook)
 * ─────────────────────────────────────────────
 *
 *   def quantize_tensor(tensor, n_bits=8):
 *       max_int = 2^(n_bits-1) − 1          # 127 for INT8
 *       absmax = max(|tensor|)
 *       scale = absmax / max_int
 *       q_tensor = clip(round(tensor / scale), −max_int, max_int).astype(int8)
 *       return q_tensor, scale, absmax
 *
 *   # Apply to all layers:
 *   W1q, S_W1, _ = quantize_tensor(W1_fused)
 *   b1q, S_b1, _ = quantize_tensor(b1_fused)
 *   W2q, S_W2, _ = quantize_tensor(W2_fused)
 *   b2q, S_b2, _ = quantize_tensor(b2_fused)
 *   W3q, S_W3, _ = quantize_tensor(W3)        # Dense layer
 *   b3q, S_b3, _ = quantize_tensor(b3)
 *
 * ACCUMULATOR BIT-WIDTH
 * ────────────────────
 *
 * During Conv MAC operations, we multiply INT8 × INT8 → INT16:
 *   product = w[i] * x[i]  (both INT8)   → INT16 (fits in 16 bits)
 *
 * We accumulate up to 180 products (for 180-tap input) or fewer (5-tap window):
 *   acc = ∑ w[i] * x[i]    (i = 0..K-1)
 *
 * Worst-case accumulator width:
 *   Suppose all weights and inputs are maxed out:
 *   max_acc = 180 × (127 × 127) ≈ 180 × 16129 ≈ 2,903,220
 *   This requires ≥ 22 bits to represent.
 *
 * Our RTL uses INT32 (32 bits) accumulators → safe headroom.
 * After accumulation, we apply scale factors and ReLU (see next section).
 *
 * FIXED-POINT ARITHMETIC ON FPGA
 * ──────────────────────────────
 *
 * The FPGA inference operates entirely in fixed-point (no floating-point):
 *
 *   1. INT8 inputs (ECG samples or previous layer outputs)
 *   2. INT8 weights and biases (quantised)
 *   3. INT32 accumulators (prevent overflow during sum)
 *   4. Apply scale factors: acc_out = (acc * S) >> shift_bits
 *   5. Clip/ReLU if needed
 *   6. Route to next layer (repeat)
 *
 * Scale factors S are converted to fixed-point shifts where possible:
 *   If S ≈ 1/2^k, then division is just a right-shift by k bits (no divider needed).
 *   Otherwise, pre-compute (S * 2^precision) as an integer multiplier.
 *
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * WORKFLOW SUMMARY
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * OFFLINE (in Python, after training):
 *   ┌─────────────────────────────────────────────────────────┐
 *   │ 1. Train model (Float32 weights & BN params)            │
 *   │ 2. Fold BN into Conv layers                             │
 *   │    → Produces W_fused, b_fused (still Float32)          │
 *   │ 3. Quantise all weights & biases to INT8               │
 *   │    → W_q, b_q, scale_q (per tensor)                    │
 *   │ 4. Export:                                              │
 *   │    a) .mem files (weight ROM initialization)            │
 *   │    b) .csv or .h files (quantisation scales)            │
 *   │    c) Verilog parameters (dimensions)                   │
 *   └─────────────────────────────────────────────────────────┘
 *
 * ONLINE (on FPGA hardware):
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ 1. Load quantised weights from BRAM                      │
 *   │ 2. Stream INT8 ECG samples (from ADC or FIFO)           │
 *   │ 3. Execute Conv → ReLU → Pool pipeline                  │
 *   │ 4. Apply scale factors as right-shifts or mults         │
 *   │ 5. Output classification (INT8 or INT16 probability)    │
 *   └──────────────────────────────────────────────────────────┘
 *
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * RTL IMPLEMENTATION NOTES
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * 1. SHIFT REGISTER (ecg_shift_reg.v)
 *    ├─ Stores last 5 ECG samples (INT8)
 *    └─ Feeds 5-tap Conv window
 *
 * 2. CONV1D_SINGLE (conv1d_single.v)
 *    ├─ 5 MACs (x[i] * w[i])
 *    ├─ Accumulation in INT32
 *    ├─ Right-shift for scaling (÷64 approximation in output)
 *    └─ No separate BN (fused into weights)
 *
 * 3. CONV1D_MULTI (conv1d_multi.v)
 *    ├─ Orchestrates 8 filters (Conv1) or 16 filters (Conv2)
 *    ├─ Loads weights from ROM on each filter
 *    └─ Cycles through filters, one output per valid pulse
 *
 * 4. RELU (relu.v)
 *    ├─ Combinational: y = max(0, x)
 *    └─ Zero latency
 *
 * 5. MAXPOOL1D (maxpool1d.v)
 *    ├─ Streaming max pooling (pool size = 2)
 *    ├─ Buffers 1st of pair, compares with 2nd
 *    └─ Outputs max every 2 input cycles
 *
 * 6. GAP_LAYER (gap_layer.v)
 *    ├─ Global average pooling (channel-wise mean)
 *    ├─ Accumulates 45 samples per channel
 *    └─ Divides by 45 (approximated as >> 5)
 *
 * 7. CLASSIFIER (Dense + Sigmoid in conv_stream_top.v)
 *    ├─ 16 × 1 matrix-vector product
 *    ├─ Bias addition
 *    └─ Sigmoid approximation (clamping to [0, 127])
 *
 * QUANTISATION SCALE APPLICATION
 * ───────────────────────────────
 *
 * In the RTL, after each MAC or layer, we apply scale factors:
 *
 *   For Conv1D output: acc is INT32
 *   ├─ Right-shift by 6 bits: acc >> 6 (approx. division by 64)
 *   │  (Note: exact S depends on trained weight distribution)
 *   ├─ Clip to INT16 range: [−32768, 32767]
 *   └─ Feed to ReLU, Pool, etc.
 *
 *   For accumulation results:
 *   ├─ Apply scale factor (shift or multiply-shift)
 *   └─ Round and saturate to output width
 *
 * PRECISION LOSS
 * ──────────────
 *
 * Sources of precision loss (in order of magnitude):
 *   1. Quantisation (largest): rounding error ±S/2 per weight
 *   2. Accumulator rounding: fixed-point round-toward-zero
 *   3. Scale factor rounding: approximating S as a power-of-2 shift
 *   4. Activation clipping (ReLU, sigmoid): negligible for normalized outputs
 *
 * Mitigation:
 *   ├─ Use INT32 accumulators (ample headroom)
 *   ├─ Apply BN folding (preserves mathematical equivalence)
 *   ├─ Post-training quantisation (measured quantisation error < 1%)
 *   └─ Validate inference accuracy before deployment
 *
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * WEIGHT FILE FORMAT (.mem files)
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * Weights are stored in Verilog .mem files (binary or hex format).
 *
 * Example: conv_weights.mem (binary format)
 *   00000001
 *   11111111  (−1 in two's complement)
 *   00101010  (42)
 *   ...
 *
 * Or hex format:
 *   01
 *   ff
 *   2a
 *   ...
 *
 * Loading in Verilog:
 *   initial $readmemb("conv_weights.mem", weights);   // binary
 *   initial $readmemh("conv_weights.mem", weights);   // hex
 *
 * Export from Python:
 *   np.savetxt("conv_weights.mem", W1q.flatten(), fmt='%d', header='')
 *   # Then manually convert to binary or use numpy formatting
 *
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * REFERENCES
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * [1] Batch Normalization: Accelerating Deep Network Training...
 *     Ioffe & Szegedy (2015)
 *     https://arxiv.org/abs/1502.03167
 *
 * [2] Fixed-Point Quantization of Deep Convolutional Networks
 *     Zhou et al. (2017)
 *     https://arxiv.org/abs/1511.04508
 *
 * [3] Quantization and Training of Neural Networks for Efficient...
 *     Jacob et al. (2018)  [TensorFlow Lite Quantization]
 *     https://arxiv.org/abs/1712.05033
 *
 * [4] Xilinx DSP48 Slice Documentation
 *     https://docs.xilinx.com/...
 *
 */
