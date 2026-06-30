use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rsomics_gaussian_filter::{FilterParams, Mode, gaussian_filter2d};

fn bench_1000x1000(c: &mut Criterion) {
    // Deterministic 1000×1000 float image, preloaded — measures compute only.
    const H: usize = 1000;
    const W: usize = 1000;
    let pixels: Vec<f64> = (0..H * W)
        .map(|i| {
            ((i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407) as f64)
                / u64::MAX as f64
        })
        .collect();

    let mut g = c.benchmark_group("gaussian_filter2d_1000x1000");

    g.bench_function("sigma1.0_nearest_float", |b| {
        let p = FilterParams {
            sigma: 1.0,
            truncate: 4.0,
            mode: Mode::Nearest,
            cval: 0.0,
            is_integer: false,
        };
        b.iter(|| gaussian_filter2d(black_box(&pixels), H, W, black_box(&p)));
    });

    g.bench_function("sigma2.5_nearest_float", |b| {
        let p = FilterParams {
            sigma: 2.5,
            truncate: 4.0,
            mode: Mode::Nearest,
            cval: 0.0,
            is_integer: false,
        };
        b.iter(|| gaussian_filter2d(black_box(&pixels), H, W, black_box(&p)));
    });

    g.bench_function("sigma5.0_nearest_float", |b| {
        let p = FilterParams {
            sigma: 5.0,
            truncate: 4.0,
            mode: Mode::Nearest,
            cval: 0.0,
            is_integer: false,
        };
        b.iter(|| gaussian_filter2d(black_box(&pixels), H, W, black_box(&p)));
    });

    g.finish();
}

criterion_group!(benches, bench_1000x1000);
criterion_main!(benches);
