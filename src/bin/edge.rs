//! Edge-test evaluator: reads cases from stdin, writes one result line per case.
//!
//! Input line:  `PREC FN KODE N NU ZR ZI [into]`
//!   PREC: f64 | f32;  FN: J Y I K H1 H2 (any N >= 1) or AI AP BI BP (N = 1)
//!   KODE: 1 = unscaled, 2 = exponentially scaled
//!   into: compute the sequence with `_seq_into` (stack workspace, long
//!         sequences in chunks) instead of `_seq`
//! Output line: `CODE NZ SINGLE RE(1) IM(1) ... RE(N) IM(N)`
//!   CODE: Fortran IERR convention (0 ok, 1 invalid input, 2 overflow,
//!         3 reduced precision, 4 total precision loss, 5 convergence failure)
//!   NZ: `underflow_count` (0 for Airy)
//!   SINGLE: 1 if N = 1 and the single-value function disagrees with the
//!           `_seq` result (value bits or error), else 0
//!
//! Bessel cases always go through `_seq` (also for N = 1) because only it
//! reports the underflow count and accuracy status.

use complex_bessel::*;
use num_complex::Complex;
use std::fmt::{LowerExp, Write as _};
use std::io::{self, BufRead, BufWriter, Write};
use std::str::FromStr;

type SeqFn<T> = fn(T, Complex<T>, usize, Scaling) -> Result<BesselResult<T>, Error>;
type SeqIntoFn<T> = fn(T, Complex<T>, Scaling, &mut [Complex<T>]) -> Result<SeqInfo, Error>;
type SingleFn<T> = fn(T, Complex<T>) -> Result<Complex<T>, Error>;
type AiryFn<T> = fn(Complex<T>, Scaling) -> Result<AiryResult<T>, Error>;

fn error_code(e: Error) -> u8 {
    match e {
        Error::InvalidInput => 1,
        Error::Overflow => 2,
        Error::TotalPrecisionLoss => 4,
        Error::ConvergenceFailure => 5,
    }
}

fn status_code(a: Accuracy) -> u8 {
    match a {
        Accuracy::Normal => 0,
        Accuracy::Reduced => 3,
    }
}

fn bessel_fns<T: BesselFloat>(name: &str, scaled: bool) -> (SeqFn<T>, SingleFn<T>) {
    match (name, scaled) {
        ("J", false) => (besselj_seq, besselj),
        ("J", true) => (besselj_seq, besselj_scaled),
        ("Y", false) => (bessely_seq, bessely),
        ("Y", true) => (bessely_seq, bessely_scaled),
        ("I", false) => (besseli_seq, besseli),
        ("I", true) => (besseli_seq, besseli_scaled),
        ("K", false) => (besselk_seq, besselk),
        ("K", true) => (besselk_seq, besselk_scaled),
        ("H1", false) => (hankel1_seq, hankel1),
        ("H1", true) => (hankel1_seq, hankel1_scaled),
        ("H2", false) => (hankel2_seq, hankel2),
        ("H2", true) => (hankel2_seq, hankel2_scaled),
        _ => panic!("unknown Bessel function {name}"),
    }
}

fn bessel_into_fn<T: BesselFloat>(name: &str) -> SeqIntoFn<T> {
    match name {
        "J" => besselj_seq_into,
        "Y" => bessely_seq_into,
        "I" => besseli_seq_into,
        "K" => besselk_seq_into,
        "H1" => hankel1_seq_into,
        "H2" => hankel2_seq_into,
        _ => panic!("unknown Bessel function {name}"),
    }
}

fn airy_fn<T: BesselFloat>(name: &str) -> Option<AiryFn<T>> {
    match name {
        "AI" => Some(airy_raw),
        "AP" => Some(airyprime_raw),
        "BI" => Some(biry_raw),
        "BP" => Some(biryprime_raw),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval<T>(
    name: &str,
    kode: u8,
    n: usize,
    nu: &str,
    zr: &str,
    zi: &str,
    into: bool,
    out: &mut String,
) where
    T: BesselFloat + BitsEq + LowerExp + FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
{
    let nu: T = nu.parse().unwrap();
    let z = Complex::new(zr.parse::<T>().unwrap(), zi.parse::<T>().unwrap());
    let scaled = kode == 2;
    let scaling = if scaled {
        Scaling::Exponential
    } else {
        Scaling::Unscaled
    };

    if let Some(f) = airy_fn::<T>(name) {
        assert_eq!(n, 1, "Airy cases must have N = 1");
        match f(z, scaling) {
            Ok(r) => write!(
                out,
                "{} 0 0 {:e} {:e}",
                status_code(r.status),
                r.value.re,
                r.value.im
            ),
            Err(e) => write!(out, "{} 0 0", error_code(e)),
        }
        .unwrap();
        return;
    }

    if into {
        let mut values = vec![Complex::new(T::zero(), T::zero()); n];
        match bessel_into_fn::<T>(name)(nu, z, scaling, &mut values) {
            Ok(info) => {
                write!(
                    out,
                    "{} {} 0",
                    status_code(info.status),
                    info.underflow_count
                )
                .unwrap();
                for v in &values {
                    write!(out, " {:e} {:e}", v.re, v.im).unwrap();
                }
            }
            Err(e) => write!(out, "{} 0 0", error_code(e)).unwrap(),
        }
        return;
    }

    let (seq, single) = bessel_fns::<T>(name, scaled);
    let res = seq(nu, z, n, scaling);
    let single_differs = n == 1
        && !match (&res, single(nu, z)) {
            (Ok(r), Ok(v)) => r.values[0].re.bits_eq(v.re) && r.values[0].im.bits_eq(v.im),
            (Err(a), Err(b)) => *a == b,
            _ => false,
        };
    match res {
        Ok(r) => {
            write!(
                out,
                "{} {} {}",
                status_code(r.status),
                r.underflow_count,
                u8::from(single_differs)
            )
            .unwrap();
            for v in &r.values {
                write!(out, " {:e} {:e}", v.re, v.im).unwrap();
            }
        }
        Err(e) => write!(out, "{} 0 {}", error_code(e), u8::from(single_differs)).unwrap(),
    }
}

/// Bitwise float equality (equal NaN payloads compare equal, +0 != -0).
trait BitsEq {
    fn bits_eq(self, other: Self) -> bool;
}

impl BitsEq for f64 {
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl BitsEq for f32 {
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

fn main() {
    let stdin = io::stdin();
    let mut w = BufWriter::new(io::stdout().lock());
    let mut out = String::new();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let f: Vec<&str> = line.split_whitespace().collect();
        let (prec, name, kode, n, nu, zr, zi, into) = match f[..] {
            [prec, name, kode, n, nu, zr, zi] => (prec, name, kode, n, nu, zr, zi, false),
            [prec, name, kode, n, nu, zr, zi, "into"] => (prec, name, kode, n, nu, zr, zi, true),
            _ => panic!("bad input line: {line}"),
        };
        let kode: u8 = kode.parse().unwrap();
        let n: usize = n.parse().unwrap();
        out.clear();
        match prec {
            "f64" => eval::<f64>(name, kode, n, nu, zr, zi, into, &mut out),
            "f32" => eval::<f32>(name, kode, n, nu, zr, zi, into, &mut out),
            _ => panic!("unknown precision {prec}"),
        }
        writeln!(w, "{out}").unwrap();
    }
}
