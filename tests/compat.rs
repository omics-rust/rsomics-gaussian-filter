// Compat tests: compare rsomics-gaussian-filter output against golden files
// produced by real skimage.filters.gaussian (scikit-image 0.26.0, scipy 1.17.1).
// Tolerance ≤1e-12 per pixel (within 1 ULP for f64 accumulation).

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn binary() -> std::path::PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("rsomics-gaussian-filter");
    p
}

struct Case {
    name: &'static str,
    sigma: f64,
    truncate: f64,
    mode: &'static str,
    cval: f64,
}

fn run_case(c: &Case) {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let in_path = golden_dir.join(format!("{}.in", c.name));
    let out_path = golden_dir.join(format!("{}.out", c.name));

    let input = std::fs::read_to_string(&in_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", in_path.display()));
    let expected_str = std::fs::read_to_string(&out_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", out_path.display()));

    let bin = binary();
    let mut child = Command::new(&bin)
        .args([
            "--sigma",
            &c.sigma.to_string(),
            "--truncate",
            &c.truncate.to_string(),
            "--mode",
            c.mode,
            "--cval",
            &c.cval.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn failed for {}: {e}", bin.display()));

    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .expect("write to stdin");

    let out = child.wait_with_output().expect("wait failed");
    assert!(
        out.status.success(),
        "binary exited {}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    let actual_str = String::from_utf8(out.stdout).expect("non-utf8 output");

    let expected: Vec<f64> = expected_str
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();
    let actual: Vec<f64> = actual_str
        .split_whitespace()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();

    assert_eq!(
        expected.len(),
        actual.len(),
        "pixel count mismatch for {}",
        c.name
    );

    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        assert!(
            diff <= 1e-12,
            "case {}: pixel[{}] expected {e:.17e} got {a:.17e} diff {diff:.3e}",
            c.name,
            i,
        );
    }
}

macro_rules! compat_test {
    ($fn_name:ident, $name:literal, sigma=$sigma:expr, truncate=$truncate:expr, mode=$mode:literal, cval=$cval:expr) => {
        #[test]
        fn $fn_name() {
            run_case(&Case {
                name: $name,
                sigma: $sigma,
                truncate: $truncate,
                mode: $mode,
                cval: $cval,
            });
        }
    };
}

// Integer input, varying sigmas, nearest mode
compat_test!(
    int_5x7_s08_nearest,
    "int_5x7_s08_nearest",
    sigma = 0.8,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_5x7_s10_nearest,
    "int_5x7_s10_nearest",
    sigma = 1.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_5x7_s25_nearest,
    "int_5x7_s25_nearest",
    sigma = 2.5,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_5x7_s50_nearest,
    "int_5x7_s50_nearest",
    sigma = 5.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);

// Integer input, varying modes
compat_test!(
    int_5x7_s10_reflect,
    "int_5x7_s10_reflect",
    sigma = 1.0,
    truncate = 4.0,
    mode = "reflect",
    cval = 0.0
);
compat_test!(
    int_5x7_s25_reflect,
    "int_5x7_s25_reflect",
    sigma = 2.5,
    truncate = 4.0,
    mode = "reflect",
    cval = 0.0
);
compat_test!(
    int_5x7_s10_constant,
    "int_5x7_s10_constant",
    sigma = 1.0,
    truncate = 4.0,
    mode = "constant",
    cval = 0.0
);
compat_test!(
    int_5x7_s10_constant05,
    "int_5x7_s10_constant05",
    sigma = 1.0,
    truncate = 4.0,
    mode = "constant",
    cval = 0.5
);
compat_test!(
    int_5x7_s10_mirror,
    "int_5x7_s10_mirror",
    sigma = 1.0,
    truncate = 4.0,
    mode = "mirror",
    cval = 0.0
);
compat_test!(
    int_5x7_s10_wrap,
    "int_5x7_s10_wrap",
    sigma = 1.0,
    truncate = 4.0,
    mode = "wrap",
    cval = 0.0
);

// Truncate variations
compat_test!(
    int_5x7_s10_trunc30,
    "int_5x7_s10_trunc30",
    sigma = 1.0,
    truncate = 3.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_5x7_s10_trunc50,
    "int_5x7_s10_trunc50",
    sigma = 1.0,
    truncate = 5.0,
    mode = "nearest",
    cval = 0.0
);

// Larger images
compat_test!(
    int_20x30_s10_nearest,
    "int_20x30_s10_nearest",
    sigma = 1.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_20x30_s25_nearest,
    "int_20x30_s25_nearest",
    sigma = 2.5,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_20x30_s25_reflect,
    "int_20x30_s25_reflect",
    sigma = 2.5,
    truncate = 4.0,
    mode = "reflect",
    cval = 0.0
);
compat_test!(
    int_50x60_s10_nearest,
    "int_50x60_s10_nearest",
    sigma = 1.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_50x60_s25_nearest,
    "int_50x60_s25_nearest",
    sigma = 2.5,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);

// Small image where kernel is larger than the image (stresses boundary extension)
compat_test!(
    int_3x3_s10_nearest,
    "int_3x3_s10_nearest",
    sigma = 1.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    int_3x3_s10_reflect,
    "int_3x3_s10_reflect",
    sigma = 1.0,
    truncate = 4.0,
    mode = "reflect",
    cval = 0.0
);
compat_test!(
    int_3x3_s10_wrap,
    "int_3x3_s10_wrap",
    sigma = 1.0,
    truncate = 4.0,
    mode = "wrap",
    cval = 0.0
);

// Float input (no /255 scaling — passthrough)
compat_test!(
    float_5x7_s08_nearest,
    "float_5x7_s08_nearest",
    sigma = 0.8,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    float_5x7_s10_nearest,
    "float_5x7_s10_nearest",
    sigma = 1.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    float_5x7_s25_nearest,
    "float_5x7_s25_nearest",
    sigma = 2.5,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    float_5x7_s50_nearest,
    "float_5x7_s50_nearest",
    sigma = 5.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);
compat_test!(
    float_5x7_s10_reflect,
    "float_5x7_s10_reflect",
    sigma = 1.0,
    truncate = 4.0,
    mode = "reflect",
    cval = 0.0
);
compat_test!(
    float_5x7_s10_constant,
    "float_5x7_s10_constant",
    sigma = 1.0,
    truncate = 4.0,
    mode = "constant",
    cval = 0.0
);
compat_test!(
    float_5x7_s10_wrap,
    "float_5x7_s10_wrap",
    sigma = 1.0,
    truncate = 4.0,
    mode = "wrap",
    cval = 0.0
);
compat_test!(
    float_20x30_s25_reflect,
    "float_20x30_s25_reflect",
    sigma = 2.5,
    truncate = 4.0,
    mode = "reflect",
    cval = 0.0
);
compat_test!(
    float_20x30_s50_nearest,
    "float_20x30_s50_nearest",
    sigma = 5.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);

// Degenerate sigma=0 → scipy passes the input straight through (identity).
// The golden .out is byte-for-byte the .in.
compat_test!(
    float_4x5_s00_nearest,
    "float_4x5_s00_nearest",
    sigma = 0.0,
    truncate = 4.0,
    mode = "nearest",
    cval = 0.0
);

// Size-1 axes under mirror (period 2n-2 = 0 for n=1). scipy handles them; a
// size-1 axis maps every offset onto its sole element.
compat_test!(
    float_1x9_s10_mirror,
    "float_1x9_s10_mirror",
    sigma = 1.0,
    truncate = 4.0,
    mode = "mirror",
    cval = 0.0
);
compat_test!(
    float_9x1_s10_mirror,
    "float_9x1_s10_mirror",
    sigma = 1.0,
    truncate = 4.0,
    mode = "mirror",
    cval = 0.0
);
compat_test!(
    float_1x1_s10_mirror,
    "float_1x1_s10_mirror",
    sigma = 1.0,
    truncate = 4.0,
    mode = "mirror",
    cval = 0.0
);
