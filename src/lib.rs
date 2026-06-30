//! Gaussian blur: separable 1-D convolution matching scipy.ndimage.gaussian_filter
// (order=0).  Two passes — rows then columns — each split into a branch-free
// interior loop and thin boundary margins of width `radius`.

/// Boundary-extension modes, matching scipy.ndimage conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Reflect,
    Constant,
    Nearest,
    Mirror,
    Wrap,
}

/// Parameters for a 2-D Gaussian filter pass.
pub struct FilterParams {
    pub sigma: f64,
    pub truncate: f64,
    pub mode: Mode,
    pub cval: f64,
    /// True when every input pixel had zero fractional part (uint8 convention).
    pub is_integer: bool,
}

/// Compute the 1-D Gaussian kernel for the given sigma and truncate.
///
/// radius = int(truncate * sigma + 0.5)
/// phi[i] = exp(-0.5 / sigma^2 * (i - radius)^2)   for i in 0..=2*radius
/// phi   /= phi.sum()
pub fn gaussian_kernel1d(sigma: f64, truncate: f64) -> Vec<f64> {
    let radius = (truncate * sigma + 0.5) as usize;
    let n = 2 * radius + 1;
    let mut phi = Vec::with_capacity(n);
    let s2 = sigma * sigma;
    for i in 0..n {
        let x = (i as f64) - (radius as f64);
        phi.push((-0.5 / s2 * x * x).exp());
    }
    // left-to-right sum, exactly as numpy does
    let sum: f64 = phi.iter().copied().sum();
    for v in &mut phi {
        *v /= sum;
    }
    phi
}

/// Map a source index with boundary extension; `n` is the axis length.
pub fn extend_index(i: isize, n: isize, mode: Mode, cval: f64, buf: &[f64]) -> f64 {
    if i >= 0 && i < n {
        return buf[i as usize];
    }
    match mode {
        Mode::Constant => cval,
        Mode::Nearest => {
            let clamped = i.clamp(0, n - 1) as usize;
            buf[clamped]
        }
        Mode::Reflect => {
            // scipy reflect: ...b a | a b c ... (edge IS repeated; period = 2n)
            let period = 2 * n;
            let mut j = ((i % period) + period) % period;
            if j >= n {
                j = 2 * n - 1 - j;
            }
            buf[j as usize]
        }
        Mode::Mirror => {
            // scipy mirror: ...c b a | b c d ... (edge not repeated; period = 2n-2)
            let period = 2 * n - 2;
            let mut j = ((i % period) + period) % period;
            if j >= n {
                j = 2 * n - 2 - j;
            }
            buf[j as usize]
        }
        Mode::Wrap => {
            let j = ((i % n) + n) % n;
            buf[j as usize]
        }
    }
}

/// Horizontal 1-D Gaussian convolution (axis=1) across all rows.
///
/// Interior columns use a tap-outer / pixel-inner loop that auto-vectorizes.
/// Boundary margins (width = radius each side) use `extend_index`.
// col is used for boundary arithmetic (si = col ± radius), not just indexing dst_row
#[allow(clippy::needless_range_loop)]
fn convolve_rows(
    src: &[f64],
    dst: &mut [f64],
    h: usize,
    w: usize,
    kernel: &[f64],
    mode: Mode,
    cval: f64,
) {
    let radius = (kernel.len() - 1) / 2;
    let interior_start = radius;
    let interior_end = w.saturating_sub(radius);

    for row in 0..h {
        let base = row * w;
        let src_row = &src[base..base + w];
        let dst_row = &mut dst[base..base + w];

        // boundary: left margin
        for col in 0..interior_start.min(w) {
            let mut acc = 0.0_f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let si = col as isize - radius as isize + ki as isize;
                acc += kv * extend_index(si, w as isize, mode, cval, src_row);
            }
            dst_row[col] = acc;
        }

        // interior: symmetric-fold, tap-outer / pixel-inner. The Gaussian kernel is
        // symmetric, so a pair of mirror taps shares one weight: center tap seeds the
        // accumulator, then pairs add inner→outer as `kv*(left+right)` — half the passes
        // over `dst`, and the exact summation order scipy's symmetric correlate1d uses.
        if interior_end > interior_start {
            let int_len = interior_end - interior_start;
            let kc = kernel[radius];
            for (d, &s) in dst_row[interior_start..interior_end]
                .iter_mut()
                .zip(&src_row[interior_start..interior_end])
            {
                *d = kc * s;
            }
            for k in 1..=radius {
                let kv = kernel[radius - k];
                let lsrc = &src_row[interior_start - k..interior_start - k + int_len];
                let rsrc = &src_row[interior_start + k..interior_start + k + int_len];
                let dst_int = &mut dst_row[interior_start..interior_end];
                for ((d, &ls), &rs) in dst_int.iter_mut().zip(lsrc).zip(rsrc) {
                    *d += kv * (ls + rs);
                }
            }
        }

        // boundary: right margin
        for col in interior_end.max(interior_start.min(w))..w {
            let mut acc = 0.0_f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let si = col as isize - radius as isize + ki as isize;
                acc += kv * extend_index(si, w as isize, mode, cval, src_row);
            }
            dst_row[col] = acc;
        }
    }
}

/// Vertical 1-D Gaussian convolution (axis=0) across all columns.
///
/// Processes columns in a column-buffer loop; interior rows use tap-outer
/// / pixel-inner to allow vectorization.
#[allow(clippy::needless_range_loop)]
fn convolve_cols(
    src: &[f64],
    dst: &mut [f64],
    h: usize,
    w: usize,
    kernel: &[f64],
    mode: Mode,
    cval: f64,
) {
    let radius = (kernel.len() - 1) / 2;
    let interior_start = radius;
    let interior_end = h.saturating_sub(radius);

    let mut col_buf = vec![0.0_f64; h];
    let mut out_buf = vec![0.0_f64; h];

    for col in 0..w {
        // gather column
        for row in 0..h {
            col_buf[row] = src[row * w + col];
        }

        // top boundary
        let top_end = interior_start.min(h);
        for row in 0..top_end {
            let mut acc = 0.0_f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let si = row as isize - radius as isize + ki as isize;
                acc += kv * extend_index(si, h as isize, mode, cval, &col_buf);
            }
            out_buf[row] = acc;
        }

        // interior: symmetric-fold (see convolve_rows) — half the passes over `out_buf`.
        if interior_end > interior_start {
            let int_len = interior_end - interior_start;
            let kc = kernel[radius];
            for (d, &s) in out_buf[interior_start..interior_end]
                .iter_mut()
                .zip(&col_buf[interior_start..interior_end])
            {
                *d = kc * s;
            }
            for k in 1..=radius {
                let kv = kernel[radius - k];
                let lsrc = &col_buf[interior_start - k..interior_start - k + int_len];
                let rsrc = &col_buf[interior_start + k..interior_start + k + int_len];
                let out_int = &mut out_buf[interior_start..interior_end];
                for ((d, &ls), &rs) in out_int.iter_mut().zip(lsrc).zip(rsrc) {
                    *d += kv * (ls + rs);
                }
            }
        }

        // bottom boundary
        let bot_start = interior_end.max(top_end);
        for row in bot_start..h {
            let mut acc = 0.0_f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let si = row as isize - radius as isize + ki as isize;
                acc += kv * extend_index(si, h as isize, mode, cval, &col_buf);
            }
            out_buf[row] = acc;
        }

        // scatter back
        for row in 0..h {
            dst[row * w + col] = out_buf[row];
        }
    }
}

/// Full 2-D Gaussian filter: two 1-D passes (axis=0 then axis=1).
///
/// If `params.is_integer` is true, divide by 255.0 first — matching
/// skimage's `img_as_float` uint8 convention.
pub fn gaussian_filter2d(pixels: &[f64], h: usize, w: usize, p: &FilterParams) -> Vec<f64> {
    let kernel = gaussian_kernel1d(p.sigma, p.truncate);

    // Scale as skimage does: integer input → divide by 255.0
    let input: Vec<f64> = if p.is_integer {
        pixels.iter().map(|&v| v / 255.0).collect()
    } else {
        pixels.to_vec()
    };

    let mut scratch = vec![0.0_f64; h * w];
    let mut output = vec![0.0_f64; h * w];

    // axis 0 first (rows), then axis 1 (cols) — same order scipy uses
    convolve_cols(&input, &mut scratch, h, w, &kernel, p.mode, p.cval);
    convolve_rows(&scratch, &mut output, h, w, &kernel, p.mode, p.cval);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sums_to_one() {
        for &sigma in &[0.5_f64, 0.8, 1.0, 2.5, 5.0] {
            let k = gaussian_kernel1d(sigma, 4.0);
            let s: f64 = k.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "sigma={sigma} sum={s}");
        }
    }

    #[test]
    fn kernel_radius_formula() {
        // radius = int(truncate * sigma + 0.5)
        assert_eq!((4.0_f64 * 1.0_f64 + 0.5) as usize, 4);
        assert_eq!((4.0_f64 * 2.5_f64 + 0.5) as usize, 10);
        assert_eq!((3.0_f64 * 1.0_f64 + 0.5) as usize, 3);
        // kernel length = 2*radius+1
        let k = gaussian_kernel1d(1.0, 4.0);
        assert_eq!(k.len(), 9, "sigma=1.0 truncate=4.0 → radius=4 → len=9");
        let k2 = gaussian_kernel1d(2.5, 4.0);
        assert_eq!(k2.len(), 21, "sigma=2.5 truncate=4.0 → radius=10 → len=21");
    }

    #[test]
    fn integer_scaling_div255() {
        // A uniform image of value 128 (integer) blurred with any symmetric
        // kernel must equal 128/255 everywhere.
        let pixels: Vec<f64> = vec![128.0; 6 * 6];
        let p = FilterParams {
            sigma: 1.0,
            truncate: 4.0,
            mode: Mode::Nearest,
            cval: 0.0,
            is_integer: true,
        };
        let out = gaussian_filter2d(&pixels, 6, 6, &p);
        let expected = 128.0 / 255.0;
        for &v in &out {
            assert!(
                (v - expected).abs() < 1e-15,
                "got {v:.17e} expected {expected:.17e}"
            );
        }
    }

    #[test]
    fn float_passthrough_no_div255() {
        // A uniform float image of value 0.5 must remain 0.5 after blur.
        let pixels: Vec<f64> = vec![0.5; 6 * 6];
        let p = FilterParams {
            sigma: 1.0,
            truncate: 4.0,
            mode: Mode::Nearest,
            cval: 0.0,
            is_integer: false,
        };
        let out = gaussian_filter2d(&pixels, 6, 6, &p);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-15, "got {v:.17e}");
        }
    }

    #[test]
    fn reflect_index_matches_scipy() {
        // arr=[0,1,2,3,4], n=5
        // scipy reflect (period=2n=10):
        //   index -1 → j=9 → >=5 → 2*5-1-9=0 → arr[0]=0
        //   index -2 → j=8 → >=5 → 2*5-1-8=1 → arr[1]=1
        //   index  5 → j=5 → >=5 → 2*5-1-5=4 → arr[4]=4
        let buf = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(extend_index(-1, 5, Mode::Reflect, 0.0, &buf), 0.0);
        assert_eq!(extend_index(-2, 5, Mode::Reflect, 0.0, &buf), 1.0);
        assert_eq!(extend_index(5, 5, Mode::Reflect, 0.0, &buf), 4.0);
        assert_eq!(extend_index(6, 5, Mode::Reflect, 0.0, &buf), 3.0);
    }

    #[test]
    fn mirror_index_matches_scipy() {
        // arr=[0,1,2,3,4], n=5
        // scipy mirror (period=2n-2=8):
        //   index -1 → j=7 → >=5 → 2*5-2-7=1 → arr[1]=1
        //   index -2 → j=6 → >=5 → 2*5-2-6=2 → arr[2]=2
        //   index  5 → j=5 → >=5 → 2*5-2-5=3 → arr[3]=3
        //   index  6 → j=6 → >=5 → 2*5-2-6=2 → arr[2]=2
        let buf = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(extend_index(-1, 5, Mode::Mirror, 0.0, &buf), 1.0);
        assert_eq!(extend_index(-2, 5, Mode::Mirror, 0.0, &buf), 2.0);
        assert_eq!(extend_index(5, 5, Mode::Mirror, 0.0, &buf), 3.0);
        assert_eq!(extend_index(6, 5, Mode::Mirror, 0.0, &buf), 2.0);
    }
}
