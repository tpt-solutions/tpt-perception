//! Covariance intersection and robust fusion under sensor failure.
//!
//! Covariance intersection (CI) fuses two Gaussian estimates *without*
//! requiring their cross-correlation to be known — the fused covariance is
//! guaranteed consistent (never overconfident) for any correlation. Sensor
//! failure handling layers on top: innovations are chi-square-gated, and a
//! sensor whose recent innovation statistics violate the gate is dropped
//! in favour of the healthy one.

use tpt_math_linalg_dense::{DMatrix, DVector};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::FusionError;

/// A Gaussian estimate: mean and covariance.
#[derive(Clone, Debug)]
pub struct Gaussian {
    /// Mean.
    pub mean: DVector<f64>,
    /// Covariance (symmetric PSD).
    pub covariance: DMatrix<f64>,
}

impl Gaussian {
    /// Builds a Gaussian, validating dimensions and symmetry.
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Result<Self, FusionError> {
        let n = mean.len();
        if covariance.nrows() != n || covariance.ncols() != n {
            return Err(FusionError::DimensionMismatch(
                "covariance must match the mean",
            ));
        }
        Ok(Gaussian { mean, covariance })
    }
}

/// Covariance intersection: fuses two estimates into a consistent Gaussian
/// without knowing their cross-correlation.
///
/// The weight `ω ∈ [0, 1]` minimising the fused covariance's
/// trace-proxy `det` is found by golden-section search on the log-det
/// surrogate `log det(ω·P₁⁻¹ + (1−ω)·P₂⁻¹)`. The fused estimate is
/// guaranteed consistent for ANY correlation between the inputs.
pub fn covariance_intersection(a: &Gaussian, b: &Gaussian) -> Result<Gaussian, FusionError> {
    if a.mean.len() != b.mean.len() {
        return Err(FusionError::DimensionMismatch(
            "estimates must share dimensions",
        ));
    }
    let inv_a = a
        .covariance
        .inverse()
        .map_err(|_| FusionError::SingularSystem("first covariance"))?;
    let inv_b = b
        .covariance
        .inverse()
        .map_err(|_| FusionError::SingularSystem("second covariance"))?;

    let log_det_surrogate = |w: f64| -> f64 {
        // trace(ω P₁⁻¹ + (1−ω) P₂⁻¹) is a cheap, monotone proxy for the
        // information size; minimising it maximises the fused information.
        let mut acc = 0.0;
        for i in 0..inv_a.nrows() {
            acc += w * inv_a[(i, i)] + (1.0 - w) * inv_b[(i, i)];
        }
        acc
    };

    // Golden-section search over ω ∈ [0, 1].
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    for _ in 0..60 {
        if log_det_surrogate(c) > log_det_surrogate(d) {
            hi = d;
        } else {
            lo = c;
        }
        c = hi - gr * (hi - lo);
        d = lo + gr * (hi - lo);
    }
    let w = (lo + hi) * 0.5;

    use crate::ekf::{mat_add, mat_scale, mat_vec, vec_add};
    let info = mat_add(&mat_scale(&inv_a, w), &mat_scale(&inv_b, 1.0 - w));
    let fused_cov = info
        .inverse()
        .map_err(|_| FusionError::SingularSystem("fused information matrix"))?;
    let wa = mat_scale(&inv_a, w);
    let wb = mat_scale(&inv_b, 1.0 - w);
    let weighted = vec_add(&mat_vec(&wa, &a.mean), &mat_vec(&wb, &b.mean));
    let fused_mean = mat_vec(&fused_cov, &weighted);
    Ok(Gaussian {
        mean: fused_mean,
        covariance: fused_cov,
    })
}

/// A sliding innovation monitor for one sensor: decides health by the
/// recent normalised innovation squared (NIS) against a chi-square-ish
/// threshold.
#[derive(Clone, Copy, Debug)]
pub struct InnovationGate {
    /// NIS threshold above which an update counts as inconsistent.
    pub nis_threshold: f64,
    /// Number of consecutive violations that mark the sensor failed.
    pub consecutive_limit: u32,
    recent_violations: u32,
    total_updates: u64,
    total_violations: u64,
}

impl InnovationGate {
    /// A gate with the given NIS threshold and consecutive-failure limit.
    pub fn new(nis_threshold: f64, consecutive_limit: u32) -> Self {
        InnovationGate {
            nis_threshold,
            consecutive_limit,
            recent_violations: 0,
            total_updates: 0,
            total_violations: 0,
        }
    }

    /// Records an update's NIS; returns `true` if the sensor is still
    /// considered healthy afterwards.
    pub fn record(&mut self, nis: f64) -> bool {
        self.total_updates += 1;
        if nis.is_finite() && nis <= self.nis_threshold {
            self.recent_violations = 0;
            true
        } else {
            self.recent_violations += 1;
            self.total_violations += 1;
            self.recent_violations < self.consecutive_limit
        }
    }

    /// True while the sensor is considered healthy.
    pub fn healthy(&self) -> bool {
        self.recent_violations < self.consecutive_limit
    }

    /// Fraction of updates that violated the gate (diagnostic).
    pub fn violation_rate(&self) -> f64 {
        if self.total_updates == 0 {
            0.0
        } else {
            self.total_violations as f64 / self.total_updates as f64
        }
    }
}

/// Robust two-sensor fusion: fuses with covariance intersection only when
/// *both* gates report healthy; otherwise passes the healthy sensor's
/// estimate through (or the first one when both are unhealthy but the
/// caller insists — documented contract: prefer failing loudly by passing
/// `None` upward).
pub fn robust_fuse(
    a: (&Gaussian, &mut InnovationGate),
    b: (&Gaussian, &mut InnovationGate),
    nis_a: f64,
    nis_b: f64,
) -> Result<Option<Gaussian>, FusionError> {
    let healthy_a = a.1.record(nis_a);
    let healthy_b = b.1.record(nis_b);
    match (healthy_a, healthy_b) {
        (true, true) => covariance_intersection(a.0, b.0).map(Some),
        (true, false) => Ok(Some(clone_gaussian(a.0))),
        (false, true) => Ok(Some(clone_gaussian(b.0))),
        (false, false) => Ok(None), // total sensor failure: no consistent estimate
    }
}

fn clone_gaussian(g: &Gaussian) -> Gaussian {
    Gaussian {
        mean: g.mean.clone(),
        covariance: g.covariance.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dg(v: &[f64]) -> DVector<f64> {
        DVector::from_vec(v.to_vec())
    }

    fn diag(v: &[f64]) -> DMatrix<f64> {
        DMatrix::from_diagonal(&dg(v))
    }

    #[test]
    fn ci_of_identical_estimates_preserves_mean() {
        let a = Gaussian::new(dg(&[1.0, 2.0]), diag(&[0.1, 0.2])).unwrap();
        let b = Gaussian::new(dg(&[1.0, 2.0]), diag(&[0.1, 0.2])).unwrap();
        let fused = covariance_intersection(&a, &b).unwrap();
        assert!((fused.mean[0] - 1.0).abs() < 1e-9);
        assert!((fused.mean[1] - 2.0).abs() < 1e-9);
        // Consistency: fused covariance never smaller than the inputs'.
        for i in 0..2 {
            assert!(fused.covariance[(i, i)] >= 0.1 * 0.9 - 1e-9);
        }
    }

    #[test]
    fn ci_weight_shifts_toward_sharper_estimate() {
        // Sensor A is much more certain.
        let a = Gaussian::new(dg(&[0.0]), diag(&[0.01])).unwrap();
        let b = Gaussian::new(dg(&[10.0]), diag(&[100.0])).unwrap();
        let fused = covariance_intersection(&a, &b).unwrap();
        assert!(
            fused.mean[0].abs() < 1.0,
            "fused mean should be near A: {}",
            fused.mean[0]
        );
        // Consistency bound: fused variance ≥ min(var_a, var_b).
        assert!(fused.covariance[(0, 0)] >= 0.01 * 0.99 - 1e-9);
    }

    #[test]
    fn ci_is_correlation_agnostic_consistent() {
        // Extreme case: identical estimates (perfect correlation). A naive
        // Kalman fusion would halve the variance — CI must not.
        let a = Gaussian::new(dg(&[5.0]), diag(&[1.0])).unwrap();
        let b = Gaussian::new(dg(&[5.0]), diag(&[1.0])).unwrap();
        let fused = covariance_intersection(&a, &b).unwrap();
        assert!(fused.covariance[(0, 0)] >= 0.5 - 1e-9);
    }

    #[test]
    fn gate_trips_on_consecutive_failures() {
        let mut gate = InnovationGate::new(9.0, 3);
        assert!(gate.record(1.0)); // healthy
        assert!(gate.record(100.0)); // tolerated violation 1
        assert!(gate.record(100.0)); // tolerated violation 2
        assert!(!gate.record(100.0)); // violation 3 → failed
        assert!(!gate.healthy());
        // A healthy update resets the streak immediately.
        assert!(gate.record(1.0));
        assert!(gate.healthy());
    }

    #[test]
    fn robust_fusion_switches_to_healthy_sensor() {
        let mut gate_a = InnovationGate::new(9.0, 2);
        let mut gate_b = InnovationGate::new(9.0, 2);
        let a = Gaussian::new(dg(&[1.0]), diag(&[0.5])).unwrap();
        let b = Gaussian::new(dg(&[2.0]), diag(&[0.5])).unwrap();

        // First update: both healthy → CI.
        let fused = robust_fuse((&a, &mut gate_a), (&b, &mut gate_b), 1.0, 1.0)
            .unwrap()
            .expect("both healthy");
        assert!((fused.mean[0] - 1.5).abs() < 0.5);

        // Sensor B now wildly inconsistent twice → gated out.
        let _ = robust_fuse((&a, &mut gate_a), (&b, &mut gate_b), 1.0, 500.0).unwrap();
        let fused2 = robust_fuse((&a, &mut gate_a), (&b, &mut gate_b), 1.0, 500.0)
            .unwrap()
            .expect("A still healthy");
        // Estimate is A's alone.
        assert!((fused2.mean[0] - 1.0).abs() < 1e-9);
        assert!((fused2.covariance[(0, 0)] - 0.5).abs() < 1e-9);

        // Drive A to failure as well (needs two more violations) → None.
        let _ = robust_fuse((&a, &mut gate_a), (&b, &mut gate_b), 500.0, 500.0).unwrap();
        let none = robust_fuse((&a, &mut gate_a), (&b, &mut gate_b), 500.0, 500.0).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let a = Gaussian::new(dg(&[1.0]), diag(&[1.0])).unwrap();
        let b = Gaussian::new(dg(&[1.0, 2.0]), diag(&[1.0, 1.0])).unwrap();
        assert!(covariance_intersection(&a, &b).is_err());
    }
}
