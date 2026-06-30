# rsomics-gaussian-filter

Gaussian blur — a value-exact Rust port of `skimage.filters.gaussian` /
`scipy.ndimage.gaussian_filter`.

## Usage

```
rsomics-gaussian-filter [OPTIONS]
```

Reads from stdin: first line `H W`, then H rows of W whitespace-separated
numbers. Integer-valued inputs are divided by 255.0 (matching skimage's
`img_as_float` uint8 convention). Outputs the blurred H×W image as
tab-separated f64 rows.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma <f64>` | `1.0` | Standard deviation (applied to both axes) |
| `--truncate <f64>` | `4.0` | Truncate filter at this many standard deviations |
| `--mode <mode>` | `nearest` | Boundary extension: `reflect`, `constant`, `nearest`, `mirror`, `wrap` |
| `--cval <f64>` | `0.0` | Fill value for `--mode constant` |

## Algorithm

Two separable 1-D Gaussian convolutions (axis 0 then axis 1), exactly
matching `scipy.ndimage.gaussian_filter` (order=0):

1. Integer input (all pixels have zero fractional part) → divide by 255.0.
2. Kernel: `radius = int(truncate * sigma + 0.5)`; `phi[i] = exp(-0.5/σ² * (i-radius)²)`,
   normalised by left-to-right sum.
3. Boundary modes match scipy exactly (verified against black-box output).

Interior pixels use a tap-outer / pixel-inner loop that auto-vectorizes.
Boundary margins of width `radius` use explicit boundary extension.

## Performance

Compute-only on a 1000×1000 image (single-threaded, M2, `aarch64-apple-darwin`):

| sigma | ours (ms) | scipy (ms) | ratio |
|-------|-----------|------------|-------|
| 1.0 | 4.69 | 5.9 | 1.26× |
| 2.5 | 8.75 | 10.0 | 1.14× |
| 5.0 | 16.4 | 16.6 | 1.01× |

Upstream: scikit-image 0.26.0, scipy 1.17.1.

## Origin

This crate is an independent Rust reimplementation based on:

- scikit-image `skimage.filters.gaussian` (BSD-3-Clause):
  <https://github.com/scikit-image/scikit-image>
- SciPy `scipy.ndimage.gaussian_filter` (BSD-3-Clause):
  <https://github.com/scipy/scipy>

Both are MIT/Apache-compatible BSD-3-Clause. The Rust implementation
reads and cites the upstream Python source directly (clean-room is not
required for BSD upstreams). No algorithmic changes — the goal is
bit-exact output with improved throughput.

License: MIT OR Apache-2.0.
