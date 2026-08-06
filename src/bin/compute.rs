use complex_bessel::*;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read};

#[derive(Deserialize)]
struct TestGrid {
    unscaled_grid: GridSpec,
    scaled_grid: GridSpec,
}

#[derive(Deserialize)]
struct GridSpec {
    nu_values: Vec<f64>,
    re_values: Vec<f64>,
    im_values: Vec<f64>,
    functions: Vec<String>,
}

#[derive(Serialize)]
struct ResultRecord {
    grid: String,
    function: String,
    idx: usize,
    nu: f64,
    z_re: f64,
    z_im: f64,
    re: Option<f64>,
    im: Option<f64>,
    status: String,
    precision: Option<String>,
}

/// Evaluate `fname` in precision `T`. Inputs are given (and outputs returned)
/// as f64; for f32 they are rounded on the way in.
fn compute_function<T: BesselFloat + Into<f64>>(
    fname: &str,
    nu: f64,
    z: Complex<f64>,
) -> (Option<f64>, Option<f64>, String, Option<String>) {
    let nu = T::from_f64(nu);
    let z = Complex::new(T::from_f64(z.re), T::from_f64(z.im));
    // Airy _raw functions: return AiryResult with Accuracy status
    if let Some(raw_result) = match fname {
        "airy" => Some(airy_raw(z, Scaling::Unscaled)),
        "airyprime" => Some(airyprime_raw(z, Scaling::Unscaled)),
        "biry" => Some(biry_raw(z, Scaling::Unscaled)),
        "biryprime" => Some(biryprime_raw(z, Scaling::Unscaled)),
        "airy_scaled" => Some(airy_raw(z, Scaling::Exponential)),
        "airyprime_scaled" => Some(airyprime_raw(z, Scaling::Exponential)),
        "biry_scaled" => Some(biry_raw(z, Scaling::Exponential)),
        "biryprime_scaled" => Some(biryprime_raw(z, Scaling::Exponential)),
        _ => None,
    } {
        return match raw_result {
            Ok(ar) => {
                let (status, precision) = match ar.status {
                    Accuracy::Normal => ("ok", "normal"),
                    Accuracy::Reduced => ("reduced_precision", "reduced"),
                };
                (
                    Some(ar.value.re.into()),
                    Some(ar.value.im.into()),
                    status.into(),
                    Some(precision.into()),
                )
            }
            Err(Error::InvalidInput) => (None, None, "invalid_input".into(), None),
            Err(Error::Overflow) => (None, None, "overflow".into(), None),
            Err(Error::TotalPrecisionLoss) => (None, None, "total_precision_loss".into(), None),
            Err(Error::ConvergenceFailure) => (None, None, "convergence_failure".into(), None),
        };
    }

    // Non-Airy Bessel functions
    let result = match fname {
        "besselj" => besselj(nu, z),
        "bessely" => bessely(nu, z),
        "besseli" => besseli(nu, z),
        "besselk" => besselk(nu, z),
        "hankel1" => hankel1(nu, z),
        "hankel2" => hankel2(nu, z),
        "besselj_scaled" => besselj_scaled(nu, z),
        "bessely_scaled" => bessely_scaled(nu, z),
        "besseli_scaled" => besseli_scaled(nu, z),
        "besselk_scaled" => besselk_scaled(nu, z),
        "hankel1_scaled" => hankel1_scaled(nu, z),
        "hankel2_scaled" => hankel2_scaled(nu, z),
        _ => return (None, None, "unknown_function".into(), None),
    };

    match result {
        Ok(val) => (
            Some(val.re.into()),
            Some(val.im.into()),
            "ok".into(),
            Some("normal".into()),
        ),
        Err(Error::InvalidInput) => (None, None, "invalid_input".into(), None),
        Err(Error::Overflow) => (None, None, "overflow".into(), None),
        Err(Error::TotalPrecisionLoss) => (None, None, "total_precision_loss".into(), None),
        Err(Error::ConvergenceFailure) => (None, None, "convergence_failure".into(), None),
    }
}

fn is_airy(fname: &str) -> bool {
    matches!(
        fname,
        "airy"
            | "airyprime"
            | "biry"
            | "biryprime"
            | "airy_scaled"
            | "airyprime_scaled"
            | "biry_scaled"
            | "biryprime_scaled"
    )
}

fn process_grid<T: BesselFloat + Into<f64>>(grid_name: &str, spec: &GridSpec) -> Vec<ResultRecord> {
    let mut records = Vec::new();

    // Pre-compute Airy results: cache by (z_re, z_im, function) -> result
    type AiryCacheEntry = (Option<f64>, Option<f64>, String, Option<String>);
    let mut airy_cache: HashMap<(u64, u64, String), AiryCacheEntry> = HashMap::new();

    let n_re = spec.re_values.len();
    let n_im = spec.im_values.len();

    for func in &spec.functions {
        if is_airy(func) {
            // Only compute unique z points for Airy (nu-independent)
            for &re in spec.re_values.iter() {
                for &im in spec.im_values.iter() {
                    let key = (re.to_bits(), im.to_bits(), func.clone());
                    airy_cache.entry(key).or_insert_with(|| {
                        let z = Complex::new(re, im);
                        compute_function::<T>(func, 0.0, z)
                    });
                }
            }
        }
    }

    for func in &spec.functions {
        let is_airy_fn = is_airy(func);

        for (ni, &nu) in spec.nu_values.iter().enumerate() {
            for (ri, &re) in spec.re_values.iter().enumerate() {
                for (ii, &im) in spec.im_values.iter().enumerate() {
                    let idx = ni * n_re * n_im + ri * n_im + ii;
                    let z = Complex::new(re, im);

                    let (res_re, res_im, status, precision) = if is_airy_fn {
                        let key = (re.to_bits(), im.to_bits(), func.clone());
                        airy_cache.get(&key).unwrap().clone()
                    } else {
                        compute_function::<T>(func, nu, z)
                    };

                    records.push(ResultRecord {
                        grid: grid_name.to_string(),
                        function: func.clone(),
                        idx,
                        nu,
                        z_re: re,
                        z_im: im,
                        re: res_re,
                        im: res_im,
                        status,
                        precision,
                    });
                }
            }
        }
    }

    records
}

/// A single input point (`--points` mode).
#[derive(Deserialize)]
struct Point {
    function: String,
    nu: f64,
    z_re: f64,
    z_im: f64,
}

#[derive(Serialize)]
struct PointResult {
    re: Option<f64>,
    im: Option<f64>,
    status: String,
}

/// Reads test_grid.json on stdin and writes results as JSON to stdout.
/// With `--f32`, evaluates in single precision; records keep the original
/// f64 grid values (ν, z) so they join with the reference results.
/// With `--points`, reads a JSON list of points `{function, nu, z_re, z_im}`
/// instead and writes one `{re, im, status}` per point (f64).
fn main() {
    let f32_mode = std::env::args().any(|a| a == "--f32");
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read stdin");

    if std::env::args().any(|a| a == "--points") {
        let points: Vec<Point> = serde_json::from_str(&input).expect("Failed to parse points");
        let results: Vec<PointResult> = points
            .iter()
            .map(|p| {
                let z = Complex::new(p.z_re, p.z_im);
                let (re, im, status, _) = compute_function::<f64>(&p.function, p.nu, z);
                PointResult { re, im, status }
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&results).expect("Failed to serialize")
        );
        return;
    }

    let grid: TestGrid = serde_json::from_str(&input).expect("Failed to parse test_grid.json");

    let mut all_records = Vec::new();

    if f32_mode {
        all_records.extend(process_grid::<f32>("unscaled", &grid.unscaled_grid));
        all_records.extend(process_grid::<f32>("scaled", &grid.scaled_grid));
    } else {
        all_records.extend(process_grid::<f64>("unscaled", &grid.unscaled_grid));
        all_records.extend(process_grid::<f64>("scaled", &grid.scaled_grid));
    }

    let json = serde_json::to_string(&all_records).expect("Failed to serialize results");
    println!("{}", json);
}
