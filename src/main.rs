use std::io::{self, BufRead, Write};

use clap::{Parser, ValueEnum};
use rsomics_gaussian_filter::{FilterParams, Mode, gaussian_filter2d};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliMode {
    Reflect,
    Constant,
    Nearest,
    Mirror,
    Wrap,
}

impl From<CliMode> for Mode {
    fn from(m: CliMode) -> Mode {
        match m {
            CliMode::Reflect => Mode::Reflect,
            CliMode::Constant => Mode::Constant,
            CliMode::Nearest => Mode::Nearest,
            CliMode::Mirror => Mode::Mirror,
            CliMode::Wrap => Mode::Wrap,
        }
    }
}

/// Gaussian blur matching skimage.filters.gaussian / scipy.ndimage.gaussian_filter.
///
/// Reads from stdin: first line is "H W", then H rows of W whitespace-separated
/// numbers. Integer-valued inputs are scaled by 1/255 (uint8 convention).
/// Outputs the blurred H×W image as tab-separated f64 rows.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Standard deviation (applied to both axes)
    #[arg(long, default_value_t = 1.0)]
    sigma: f64,

    /// Truncate filter at this many standard deviations
    #[arg(long, default_value_t = 4.0)]
    truncate: f64,

    /// Boundary extension mode
    #[arg(long, default_value = "nearest")]
    mode: CliMode,

    /// Constant fill value (used only with --mode constant)
    #[arg(long, default_value_t = 0.0)]
    cval: f64,
}

fn main() {
    let cli = Cli::parse();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let header = lines
        .next()
        .expect("expected H W header line")
        .expect("failed to read header");
    let mut parts = header.split_whitespace();
    let h: usize = parts
        .next()
        .expect("H missing")
        .parse()
        .expect("H not a number");
    let w: usize = parts
        .next()
        .expect("W missing")
        .parse()
        .expect("W not a number");

    let mut pixels = Vec::with_capacity(h * w);
    let mut is_integer = true;

    for _ in 0..h {
        let line = lines
            .next()
            .expect("unexpected end of input")
            .expect("failed to read line");
        for tok in line.split_whitespace() {
            let v: f64 = tok
                .parse()
                .unwrap_or_else(|_| panic!("not a number: {tok}"));
            if v.fract() != 0.0 {
                is_integer = false;
            }
            pixels.push(v);
        }
    }

    let params = FilterParams {
        sigma: cli.sigma,
        truncate: cli.truncate,
        mode: Mode::from(cli.mode),
        cval: cli.cval,
        is_integer,
    };
    let result = gaussian_filter2d(&pixels, h, w, &params);

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for row in 0..h {
        let base = row * w;
        let row_slice = &result[base..base + w];
        let mut first = true;
        for &v in row_slice {
            if !first {
                out.write_all(b"\t").unwrap();
            }
            write!(out, "{v:.17e}").unwrap();
            first = false;
        }
        out.write_all(b"\n").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
