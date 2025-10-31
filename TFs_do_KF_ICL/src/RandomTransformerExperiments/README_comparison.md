# GPT2 Pure vs Modified Comparison

## Overview

This experiment compares two versions of the pretrained GPT2 model:

1. **GPT2 Pure**: Pretrained model as-is (checkpoint at step 99,000)
2. **GPT2 Modified**: Same pretrained model but with the outer readout layer optimized via least squares on training data

## Data Structure

### Haystack Files (6 total)
- `val_interleaved_traces_ortho_haar_ident_C_haystack_len_[1-6].pkl`
- Each file contains interleaved traces with different sequence lengths
- Shape: `(50 configs, 1 trace, 1000 examples, seq_len, 57 dims)`

### Input Dimensions (57 total)
- **Indices 0-49**: Open/close tokens for 25 systems
  - Even indices (0,2,4,...,48): Open brackets for systems 0,1,2,...,24
  - Odd indices (1,3,5,...,49): Close brackets for systems 0,1,2,...,24
- **Index 50**: First element flag (nonzero only at position 0 of trace)
- **Index 51**: Payload flag (indicates observation vector)
- **Indices 52-56**: The 5-dimensional observation vector

## Training Procedure

### Data Extraction
1. Process entire trace through GPT2 to get all last-layer activations
2. For each position `i` where `trace[i+1, 51] != 0` (next token has payload):
   - **Input**: `activation[i]` (128-dim vector from last layer before readout)
   - **Target**: `trace[i+1, 52:57]` (last 5 dimensions of next token)

### Least Squares Optimization
- Collect all (activation, target) pairs into matrices A and Y
- Solve: `W = (A^T A + bias)^{-1} A^T Y`
- Update model's `_read_out` layer with new weights W and bias b

### Training Splits
- **Training**: First 40 configs (40,000 examples total)
- **Testing**: Last 10 configs (10,000 examples total)
- Progressive training with: 1, 5, 10, 20, 40 configs

## Evaluation Metrics

For each unique open bracket in a trace:

### k_after_initial
Prediction accuracy k positions after the **first occurrence** of an open bracket

### k_after_final  
Prediction accuracy k positions after the **last occurrence** of the same open bracket

### k values tested
- k = 1, 2, 3, 7, 8 (5 values)
- Total: 10 metrics per model (5 after_initial + 5 after_final)

## Plotting

### 3 Plots (one per haystack length 1, 2, 3)

Each plot contains:
- **X-axis**: Number of training configs used (1, 5, 10, 20, 40)
- **Y-axis**: MSE (log scale)
- **20 curves total**:
  - 10 for GPT2 Pure (dashed lines, flat because model doesn't change)
  - 10 for GPT2 Modified (solid lines, improve with more training data)
- **Color coding**: Matching colors for corresponding Pure/Modified pairs

### Legend Format
- Same color for corresponding metrics
- Dashed = Pure (baseline)
- Solid = Modified (learned outer layer)

## Expected Results

- **GPT2 Pure**: Flat lines (performance doesn't change)
- **GPT2 Modified**: Curves that improve (decrease) with more training data
- Modified should eventually match or beat Pure on the specific test distribution

## Files Created

1. `compare_gpt2_pure_vs_modified.py` - Main comparison script
2. `gpt2_pure_vs_modified_comparison.pdf` - Output plots
3. `gpt2_comparison_results.pkl` - Numerical results








