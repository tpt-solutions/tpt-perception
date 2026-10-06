//! Self-contained Bayesian filters: EKF and UKF over dense states.
//!
//! Both filters are generic over the process/measurement models (closures)
//! with externally supplied Jacobians (EKF) or pure functions (UKF — the
//! sigma-point transform differentiates numerically by construction).
//! This is the fusion crate's own filtering substrate per the resolved
//! checkpoint in `todo.md`: `tpt-control` does not exist yet; when it does,
//! these remain valid lightweight fallbacks while motion models migrate.
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg_dense::{DMatrix, DVector};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::FusionError;

/// Extended Kalman Filter over a dense state.
///
/// # Example
/// ```
/// use tpt_math_linalg_dense::{DMatrix, DVector};
/// use tpt_percept_fusion::ekf::Ekf;
///
/// // 1-D constant-velocity tracking.
/// let mut ekf = Ekf::new(
///     DVector::from_vec(vec![0.0, 1.0]),
///     DMatrix::from_diagonal(&DVector::from_vec(vec![0.1, 0.1])),
/// );
/// // x' = x + v·dt
/// let dt = 0.1_f64;
/// let mut f = |x: &DVector| DVector::from_vec(vec![x[0] + x[1] * dt, x[1]]);
/// let f_jac = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
/// ekf.predict(&mut f, &f_jac, &DMatrix::from_diagonal(&DVector::from_vec(vec![1e-4, 1e-4])))
///     .unwrap();
/// // Position measurement.
/// let mut h = |x: &DVector| DVector::from_vec(vec![x[0]]);
/// let h_jac = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
/// let nis = ekf
///     .update(&DVector::from_vec(vec![0.15]), &mut h, &h_jac,
///             &DMatrix::from_diagonal(&DVector::from_vec(vec![0.01])))
///     .unwrap();
/// assert!(nis.is_finite());
/// ```
#[derive(Clone, Debug)]
pub struct Ekf {
    state: DVector<f64>,
    covariance: DMatrix<f64>,
}

impl Ekf {
    /// Creates a filter with an initial state and covariance.
    pub fn new(state: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        Ekf { state, covariance }
    }

    /// Current state.
    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Current covariance.
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    /// Process step: `x ← f(x)`, `P ← F P Fᵀ + Q`.
    pub fn predict(
        &mut self,
        f: &mut dyn FnMut(&DVector<f64>) -> DVector<f64>,
        f_jacobian: &DMatrix<f64>,
        process_noise: &DMatrix<f64>,
    ) -> Result<(), FusionError> {
        let n = self.state.len();
        if f_jacobian.nrows() != n || f_jacobian.ncols() != n {
            return Err(FusionError::DimensionMismatch("F must be n×n"));
        }
        self.state = f(&self.state);
        let fp = mat_mul(f_jacobian, &self.covariance);
        let pf_t = mat_mul(&fp, &f_jacobian.transpose());
        self.covariance = mat_add(&pf_t, process_noise);
        Ok(())
    }

    /// Measurement step with chi-square innovation gating information.
    ///
    /// Returns the normalised innovation squared (NIS): `νᵀ S⁻¹ ν`. Values
    /// persistently above `dim_state` indicate model/measurements mismatch
    /// (the caller can gate on this — see [`crate::robust`]).
    pub fn update(
        &mut self,
        measurement: &DVector<f64>,
        h: &mut dyn FnMut(&DVector<f64>) -> DVector<f64>,
        h_jacobian: &DMatrix<f64>,
        measurement_noise: &DMatrix<f64>,
    ) -> Result<f64, FusionError> {
        let m = measurement.len();
        if h_jacobian.nrows() != m || h_jacobian.ncols() != self.state.len() {
            return Err(FusionError::DimensionMismatch("H must be m×n"));
        }
        let predicted = h(&self.state);
        let innovation = vec_sub(measurement, &predicted);
        let h_t = h_jacobian.transpose();
        let ph_t = mat_mul(&self.covariance, &h_t);
        let s = mat_add(&mat_mul(h_jacobian, &ph_t), measurement_noise);

        let s_inv = s
            .inverse()
            .map_err(|_| FusionError::SingularSystem("innovation covariance"))?;
        // K = P Hᵀ S⁻¹ (already n×m; S⁻¹ symmetric).
        let k = mat_mul(&ph_t, &s_inv);

        // NIS = νᵀ S⁻¹ ν.
        let nis_v = mat_vec(&s_inv, &innovation);
        let nis = innovation.dot(&nis_v);

        let correction = mat_vec(&k, &innovation);
        self.state = vec_add(&self.state, &correction);
        let kh = mat_mul(&k, h_jacobian);
        // Joseph form for numerical symmetry: P = (I−KH) P (I−KH)ᵀ + K R Kᵀ.
        let imk = mat_sub(&identity(n_of(&kh)), &kh);
        let p1 = mat_mul(&imk, &self.covariance);
        let p2 = mat_mul(&p1, &imk.transpose());
        let kr = mat_mul(&k, measurement_noise);
        let p3 = mat_mul(&kr, &k.transpose());
        self.covariance = symmetrised(&mat_add(&p2, &p3));
        Ok(nis)
    }
}

fn n_of(kh: &DMatrix<f64>) -> usize {
    kh.nrows()
}

/// Sigma points with mean- and covariance-weight vectors.
pub type SigmaPoints = (Vec<DVector<f64>>, Vec<f64>, Vec<f64>);

/// Owned-value matrix helpers (the dense backend implements operators on
/// owned values only).
pub(crate) fn identity(n: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 })
}

pub(crate) fn mat_mul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a.clone() * b.clone()
}

pub(crate) fn mat_add(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a.clone() + b.clone()
}

pub(crate) fn mat_sub(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a.clone() - b.clone()
}

pub(crate) fn mat_scale(a: &DMatrix<f64>, s: f64) -> DMatrix<f64> {
    a.clone() * s
}

pub(crate) fn vec_sub(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a.clone() - b.clone()
}

pub(crate) fn vec_add(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a.clone() + b.clone()
}

pub(crate) fn mat_vec(a: &DMatrix<f64>, v: &DVector<f64>) -> DVector<f64> {
    a.clone() * v.clone()
}

/// Outer product of two vectors (`a · bᵀ`).
pub(crate) fn vec_outer(a: &DVector<f64>) -> DMatrix<f64> {
    let n = a.len();
    DMatrix::from_fn(n, n, |i, j| a[i] * a[j])
}

/// Cross product of two vectors (`a · bᵀ`, rectangular).
pub(crate) fn vec_cross(a: &DVector<f64>, b: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(a.len(), b.len(), |i, j| a[i] * b[j])
}

/// Symmetrises a matrix through a flat buffer (DMatrix is immutable).
pub(crate) fn symmetrised(m: &DMatrix<f64>) -> DMatrix<f64> {
    let n = m.nrows();
    let mut flat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = if j < i {
                (m[(i, j)] + m[(j, i)]) * 0.5
            } else if j == i {
                m[(i, i)]
            } else {
                (m[(i, j)] + m[(j, i)]) * 0.5
            };
            flat[i * n + j] = v;
        }
    }
    DMatrix::from_vec(n, n, flat)
}

/// Unscented Kalman Filter over a dense state.
#[derive(Clone, Debug)]
pub struct Ukf {
    state: DVector<f64>,
    covariance: DMatrix<f64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
}

impl Ukf {
    /// Creates a UKF with the standard sigma-point scaling parameters.
    pub fn new(state: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        Ukf {
            state,
            covariance,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    /// Current state.
    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Current covariance.
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    /// Overrides the sigma-point scaling (α ∈ (0, 1], β ≥ 0, κ ≥ −3).
    pub fn with_scaling(mut self, alpha: f64, beta: f64, kappa: f64) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self.kappa = kappa;
        self
    }

    /// Process step through the sigma-point transform.
    pub fn predict(
        &mut self,
        f: &mut dyn FnMut(&DVector<f64>) -> DVector<f64>,
        process_noise: &DMatrix<f64>,
    ) -> Result<(), FusionError> {
        let n = self.state.len();
        let (sigma, weights_mean, _) = self.sigma_points()?;
        let mut transformed = Vec::with_capacity(2 * n + 1);
        for s in &sigma {
            transformed.push(f(s));
        }
        let mut x = DVector::zeros(n);
        for (i, s) in transformed.iter().enumerate() {
            x = vec_add(&x, &(s.clone() * weights_mean[i]));
        }
        let mut p = process_noise.clone();
        for (i, s) in transformed.iter().enumerate() {
            let d = vec_sub(s, &x);
            let w = weights_cov(i, n, self.alpha, self.beta, self.kappa);
            let outer = vec_outer(&d);
            p = mat_add(&p, &mat_scale(&outer, w));
        }
        self.state = x;
        self.covariance = p;
        Ok(())
    }

    /// Measurement step; returns the NIS of the innovation.
    pub fn update(
        &mut self,
        measurement: &DVector<f64>,
        h: &mut dyn FnMut(&DVector<f64>) -> DVector<f64>,
        measurement_noise: &DMatrix<f64>,
    ) -> Result<f64, FusionError> {
        let n = self.state.len();
        let m = measurement.len();
        let (sigma, weights_mean, weight_fn) = {
            let (s, wm, _wc) = self.sigma_points()?;
            (s, wm, _wc)
        };
        let weights_cov_fn = |i: usize| weight_fn.get(i).copied().unwrap_or(0.0);
        let mut z_sigma = Vec::with_capacity(2 * n + 1);
        for s in &sigma {
            z_sigma.push(h(s));
        }
        let mut z_pred = DVector::zeros(m);
        for (i, z) in z_sigma.iter().enumerate() {
            z_pred = vec_add(&z_pred, &(z.clone() * weights_mean[i]));
        }
        // Cross-covariance and innovation covariance.
        let mut p_xz = DMatrix::zeros(n, m);
        let mut s = measurement_noise.clone();
        for i in 0..(2 * n + 1) {
            let dx = vec_sub(&sigma[i], &self.state);
            let dz = vec_sub(&z_sigma[i], &z_pred);
            let w = weights_cov_fn(i);
            p_xz = mat_add(&p_xz, &mat_scale(&vec_cross(&dx, &dz), w));
            s = mat_add(&s, &mat_scale(&vec_outer(&dz), w));
        }
        let s_inv = s
            .inverse()
            .map_err(|_| FusionError::SingularSystem("UKF innovation covariance"))?;
        let k = mat_mul(&p_xz, &s_inv);
        let innovation = vec_sub(measurement, &z_pred);
        let nis_v = mat_vec(&s_inv, &innovation);
        let nis = innovation.dot(&nis_v);
        let correction = mat_vec(&k, &innovation);
        self.state = vec_add(&self.state, &correction);
        // P ← P − K S Kᵀ (UKF standard form).
        let ks = mat_mul(&k, &s);
        let ksk_t = mat_mul(&ks, &k.transpose());
        self.covariance = symmetrised(&mat_sub(&self.covariance, &ksk_t));
        let _ = n;
        Ok(nis)
    }

    /// Sigma points and (mean, covariance) weights.
    fn sigma_points(&self) -> Result<SigmaPoints, FusionError> {
        let n = self.state.len();
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;
        let scale = n as f64 + lambda;
        if scale <= 0.0 {
            return Err(FusionError::InvalidParameter("sigma scaling must be > 0"));
        }
        let chol = cholesky(&self.covariance)?;
        let mut sigma = Vec::with_capacity(2 * n + 1);
        sigma.push(self.state.clone());
        let root = scale.sqrt();
        for j in 0..n {
            let mut plus: Vec<f64> = (0..n).map(|i| self.state[i]).collect();
            for i in 0..n {
                plus[i] += root * chol[i][j];
            }
            sigma.push(DVector::from_vec(plus));
        }
        for j in 0..n {
            let mut minus: Vec<f64> = (0..n).map(|i| self.state[i]).collect();
            for i in 0..n {
                minus[i] -= root * chol[i][j];
            }
            sigma.push(DVector::from_vec(minus));
        }
        let mut weights_mean = Vec::with_capacity(2 * n + 1);
        let mut weights_cov = Vec::with_capacity(2 * n + 1);
        weights_mean.push(lambda / scale);
        weights_cov.push(lambda / scale + self.beta + 1.0 - self.alpha * self.alpha);
        let base = 1.0 / (2.0 * scale);
        for _ in 0..2 * n {
            weights_mean.push(base);
            weights_cov.push(base);
        }
        Ok((sigma, weights_mean, weights_cov))
    }
}

fn weights_cov(i: usize, n: usize, alpha: f64, beta: f64, kappa: f64) -> f64 {
    let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;
    if i == 0 {
        lambda / (n as f64 + lambda) + beta + 1.0 - alpha * alpha
    } else {
        1.0 / (2.0 * (n as f64 + lambda))
    }
}

/// Cholesky factorisation of a symmetric positive-definite matrix; returns
/// the lower-triangular factor as rows.
pub(crate) fn cholesky(m: &DMatrix<f64>) -> Result<[[f64; 16]; 16], FusionError> {
    let n = m.nrows();
    if n != m.ncols() {
        return Err(FusionError::DimensionMismatch(
            "Cholesky needs a square matrix",
        ));
    }
    if n > 16 {
        return Err(FusionError::InvalidParameter(
            "UKF supports states up to dimension 16",
        ));
    }
    let mut l = [[0.0f64; 16]; 16];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = m[(i, j)];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(FusionError::SingularSystem(
                        "covariance is not positive definite",
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_vec(v.to_vec())
    }

    fn diag(v: &[f64]) -> DMatrix<f64> {
        DMatrix::from_diagonal(&dv(v))
    }

    #[test]
    fn ekf_tracks_constant_velocity() {
        let mut ekf = Ekf::new(dv(&[0.0, 1.0]), diag(&[0.1, 0.1]));
        let dt = 0.1_f64;
        let f_jac = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let q = diag(&[1e-5; 2]);
        let h_jac = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r = diag(&[0.01]);

        let mut x = 0.0_f64;
        let v = 1.0_f64;
        let mut seed = 42_u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / u32::MAX as f64) - 0.5
        };
        for _ in 0..50 {
            ekf.predict(
                &mut |s: &DVector<f64>| dv(&[s[0] + s[1] * dt, s[1]]),
                &f_jac,
                &q,
            )
            .unwrap();
            x += v * dt;
            let z = x + 0.1 * rand();
            ekf.update(&dv(&[z]), &mut |s: &DVector<f64>| dv(&[s[0]]), &h_jac, &r)
                .unwrap();
        }
        let err = (ekf.state()[0] - x).abs();
        assert!(err < 0.15, "position error {err}");
        let err_v = (ekf.state()[1] - v).abs();
        assert!(err_v < 0.2, "velocity error {err_v}");
    }

    #[test]
    fn ekf_nis_detects_model_mismatch() {
        let mut ekf = Ekf::new(dv(&[0.0, 0.0]), diag(&[0.01, 0.01]));
        let f_jac = identity(2);
        let h_jac = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        ekf.predict(
            &mut |s: &DVector<f64>| dv(&[s[0], s[1]]),
            &f_jac,
            &diag(&[1e-6, 1e-6]),
        )
        .unwrap();
        // Wildly inconsistent measurement → huge NIS.
        let nis = ekf
            .update(
                &dv(&[10.0]),
                &mut |s: &DVector<f64>| dv(&[s[0]]),
                &h_jac,
                &diag(&[0.01]),
            )
            .unwrap();
        assert!(nis > 100.0, "NIS {nis}");
    }

    #[test]
    fn ukf_tracks_constant_velocity() {
        let mut ukf = Ukf::new(dv(&[0.0, 1.0]), diag(&[0.1, 0.1]));
        let dt = 0.1_f64;
        let q = diag(&[1e-5, 1e-5]);
        let r = diag(&[0.01]);
        let mut x = 0.0_f64;
        let mut seed = 7_u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / u32::MAX as f64) - 0.5
        };
        for _ in 0..50 {
            ukf.predict(&mut |s: &DVector<f64>| dv(&[s[0] + s[1] * dt, s[1]]), &q)
                .unwrap();
            x += dt;
            let z = x + 0.1 * rand();
            ukf.update(&dv(&[z]), &mut |s: &DVector<f64>| dv(&[s[0]]), &r)
                .unwrap();
        }
        let err_v = (ukf.state()[1] - 1.0).abs();
        assert!(err_v < 0.25, "ukf velocity error {err_v}");
    }

    #[test]
    fn cholesky_reconstructs() {
        let m = diag(&[4.0, 9.0, 16.0]);
        let l = cholesky(&m).unwrap();
        for i in 0..3 {
            assert!((l[i][i] - [2.0, 3.0, 4.0][i]).abs() < 1e-12);
        }
    }
}
