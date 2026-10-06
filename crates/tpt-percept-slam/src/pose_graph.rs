//! Pose graph optimization: dense Gauss–Newton over SE(3) poses with
//! numeric Jacobians.
//!
//! Nodes carry poses; edges carry *relative* pose measurements
//! `T_ij ≈ T_i⁻¹ · T_j` (odometry, loop closure). The optimiser minimises
//! `Σ ‖log(T_ij_meas⁻¹ · T_i⁻¹ T_j)‖²_{Ω}` — the SE(3) log residual with an
//! optional Huber robustifier — by dense Gauss–Newton with left-multiplied
//! perturbations.
//!
//! **Scale contract:** the solver is intentionally dense (O(n²) memory,
//! numeric Jacobians) and targets graphs of up to a few hundred nodes —
//! enough for loop-closure batches. Large-scale sparse optimization is the
//! open `tpt-systems-optimisation` checkpoint in `todo.md`; swap the solver
//! there when that crate becomes available.
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg_dense::{DMatrix, DVector};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::SlamError;

/// A pose as `(R, t)` — `p_world = R · p_body + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Rotation (proper orthonormal).
    pub rotation: [[f64; 3]; 3],
    /// Translation (metres).
    pub translation: [f64; 3],
}

impl Pose {
    /// Identity pose.
    pub fn identity() -> Self {
        Pose {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    /// Compose `self ∘ other` (apply `other` first).
    pub fn compose(&self, other: &Pose) -> Pose {
        let r = mul3(&self.rotation, &other.rotation);
        let t = add3(
            mul_mat_vec(&self.rotation, other.translation),
            self.translation,
        );
        Pose {
            rotation: r,
            translation: t,
        }
    }

    /// Inverse pose.
    pub fn inverse(&self) -> Pose {
        let r_t = transpose3(&self.rotation);
        Pose {
            rotation: r_t,
            translation: scale3(mul_mat_vec(&r_t, self.translation), -1.0),
        }
    }

    /// Left-multiplied SE(3) exponential perturbation: `self ← exp(ξ) · self`
    /// with `ξ = (v, ω)`.
    pub fn left_perturb(&self, v: &[f64; 3], w: &[f64; 3]) -> Pose {
        let theta = norm3(w);
        let dr = if theta < 1e-12 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            rodrigues(w, theta)
        };
        // Exact SE(3) exp translation (see tpt-percept-core twist::exp).
        let wdt = *w; // unit-time twist
        let wv = cross3(wdt, *v);
        let w2v = cross3(wdt, wv);
        let theta2 = theta * theta;
        let a = if theta2 > 1e-16 {
            (1.0 - theta.cos()) / theta2
        } else {
            0.5
        };
        let b = if theta2 > 1e-16 {
            (theta - theta.sin()) / (theta2 * theta)
        } else {
            1.0 / 6.0
        };
        let t = [
            v[0] + a * wv[0] + b * w2v[0],
            v[1] + a * wv[1] + b * w2v[1],
            v[2] + a * wv[2] + b * w2v[2],
        ];
        Pose {
            rotation: mul3(&dr, &self.rotation),
            translation: add3(mul_mat_vec(&dr, self.translation), t),
        }
    }
}

/// A relative-pose constraint between two nodes.
#[derive(Clone, Copy, Debug)]
pub struct PoseEdge {
    /// Source node.
    pub from: usize,
    /// Target node.
    pub to: usize,
    /// Measured relative pose `T_i⁻¹ · T_j`.
    pub relative: Pose,
    /// Information scale (weight) — scalar multiple of the identity
    /// information matrix.
    pub information: f64,
    /// Huber robustifier delta (0 disables robustification).
    pub huber_delta: f64,
}

/// Optimizer settings.
#[derive(Clone, Copy, Debug)]
pub struct PoseGraphParams {
    /// Maximum Gauss–Newton iterations.
    pub max_iterations: u32,
    /// Stop when the total cost decreases by less than this (relative).
    pub tolerance: f64,
    /// Central-difference step for numeric Jacobians.
    pub numeric_step: f64,
}

impl Default for PoseGraphParams {
    fn default() -> Self {
        PoseGraphParams {
            max_iterations: 50,
            tolerance: 1e-10,
            numeric_step: 1e-7,
        }
    }
}

/// A pose graph: nodes are poses in insertion order; edges reference node
/// indices.
#[derive(Clone, Debug, Default)]
pub struct PoseGraph {
    poses: Vec<Pose>,
    edges: Vec<PoseEdge>,
}

impl PoseGraph {
    /// An empty graph.
    pub fn new() -> Self {
        PoseGraph::default()
    }

    /// Adds a node, returning its index.
    pub fn add_node(&mut self, pose: Pose) -> usize {
        self.poses.push(pose);
        self.poses.len() - 1
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// True if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Borrows a node pose.
    pub fn pose(&self, index: usize) -> Option<&Pose> {
        self.poses.get(index)
    }

    /// Adds a constraint edge.
    pub fn add_edge(&mut self, edge: PoseEdge) -> Result<(), SlamError> {
        if edge.from >= self.poses.len() || edge.to >= self.poses.len() {
            return Err(SlamError::InvalidParameter("edge references missing node"));
        }
        if !(edge.information.is_finite() && edge.information >= 0.0) {
            return Err(SlamError::InvalidParameter(
                "information must be finite and >= 0",
            ));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Borrows an edge by index.
    pub fn edge(&self, index: usize) -> Option<&PoseEdge> {
        self.edges.get(index)
    }

    /// Total squared (Huber-weighted) constraint cost.
    pub fn cost(&self) -> f64 {
        self.edges
            .iter()
            .map(|e| {
                let r = self.residual(e);
                let mut w = e.information;
                if e.huber_delta > 0.0 {
                    let n = norm6(r);
                    if n > e.huber_delta {
                        w *= e.huber_delta / n;
                    }
                }
                w * dot6(r, r)
            })
            .sum()
    }

    /// SE(3) log residual of an edge (tangent vector, 6 components).
    fn residual(&self, edge: &PoseEdge) -> [f64; 6] {
        let ti = &self.poses[edge.from];
        let tj = &self.poses[edge.to];
        // e = log( T_ij_meas⁻¹ · T_i⁻¹ T_j ).
        let measured_inv = edge.relative.inverse();
        let actual = ti.inverse().compose(tj);
        let err = measured_inv.compose(&actual);
        se3_log(&err)
    }

    /// Runs dense Gauss–Newton optimization (node 0 is held fixed).
    pub fn optimize(&mut self, params: &PoseGraphParams) -> Result<usize, SlamError> {
        if self.poses.is_empty() {
            return Ok(0);
        }
        let n = self.poses.len();
        let mut iterations = 0usize;

        for _ in 0..params.max_iterations {
            iterations += 1;
            let dim = 6 * (n - 1); // node 0 is the fixed anchor
            let mut h_flat = vec![0.0f64; dim * dim];
            let mut g_flat = vec![0.0f64; dim];

            for edge in &self.edges {
                // Numeric Jacobian of the 6-vector residual wrt the left
                // perturbation (v, ω) of both endpoint poses:
                // jac[residual row][12 columns = (node, dof)].
                let r0 = self.residual(edge);
                let mut jac = [[0.0f64; 12]; 6];
                for k in 0..6 {
                    for (which, &node) in [edge.from, edge.to].iter().enumerate() {
                        let step = params.numeric_step;
                        let mut dir = [0.0f64; 6];
                        dir[k] = step;
                        let mut plus = self.clone();
                        let (v, w) = (&dir[0..3], &dir[3..6]);
                        plus.poses[node] =
                            self.poses[node].left_perturb(&[v[0], v[1], v[2]], &[w[0], w[1], w[2]]);
                        let rp = plus.residual(edge);
                        for row in 0..6 {
                            jac[row][which * 6 + k] = (rp[row] - r0[row]) / step;
                        }
                    }
                }

                // Robust weight (Huber on the residual norm).
                let mut weight = edge.information;
                if edge.huber_delta > 0.0 {
                    let nrm = norm6(r0);
                    if nrm > edge.huber_delta {
                        weight *= edge.huber_delta / nrm;
                    }
                }

                // Accumulate H = Σ Jᵀ W J, g = Σ Jᵀ W r in free-block
                // indexing: node 0 is anchored, free node i (≥ 1) occupies
                // rows/cols (i−1)·6 … +6.
                let global = |col: usize| -> Option<usize> {
                    let node = if col < 6 { edge.from } else { edge.to };
                    if node == 0 {
                        None
                    } else {
                        Some((node - 1) * 6 + (col % 6))
                    }
                };
                for a_ in 0..12 {
                    let Some(ga) = global(a_) else { continue };
                    for b_ in 0..12 {
                        let Some(gb) = global(b_) else { continue };
                        let mut acc = 0.0;
                        for r in 0..6 {
                            acc += jac[r][a_] * jac[r][b_];
                        }
                        h_flat[ga * dim + gb] += weight * acc;
                    }
                    let mut acc = 0.0;
                    for r in 0..6 {
                        acc += jac[r][a_] * r0[r];
                    }
                    // Gauss–Newton descent: δ = −H⁻¹ (Σ Jᵀ W r).
                    g_flat[ga] -= weight * acc;
                }
            }

            // Solve H δ = −g for the free nodes (0 is fixed).
            let h = DMatrix::from_vec(dim, dim, h_flat);
            let g = DVector::from_vec(g_flat);
            let sol = match h.solve(&g) {
                Ok(delta) => delta,
                Err(_) => break, // singular: stop
            };
            let mut max_step = 0.0f64;
            for node in 1..n {
                let base = (node - 1) * 6; // free-block index of this node
                let v = [sol[base], sol[base + 1], sol[base + 2]];
                let w = [sol[base + 3], sol[base + 4], sol[base + 5]];
                self.poses[node] = self.poses[node].left_perturb(&v, &w);
                max_step = max_step.max(norm3(&v)).max(norm3(&w));
            }
            if max_step < 1e-12 {
                break;
            }
        }
        Ok(iterations)
    }
}

/// SE(3) logarithm: tangent coordinates `(v, ω)` of `T`.
fn se3_log(t: &Pose) -> [f64; 6] {
    let r = &t.rotation;
    let tr = r[0][0] + r[1][1] + r[2][2];
    let cos_t = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_t.acos();
    let (axis, angle) = if theta < 1e-10 {
        ([0.0, 0.0, 0.0], 0.0)
    } else if (core::f64::consts::PI - theta).abs() < 1e-6 {
        let mut k = [0.0f64; 3];
        for (i, ki) in k.iter_mut().enumerate() {
            *ki = ((r[i][i] + 1.0) / 2.0).max(0.0).sqrt();
        }
        let m = if k[0] >= k[1] && k[0] >= k[2] {
            0
        } else if k[1] >= k[2] {
            1
        } else {
            2
        };
        if k[m] > 1e-12 {
            let km = k[m];
            for j in 0..3 {
                if j != m {
                    k[j] = r[m][j] / (2.0 * km);
                }
            }
            (k, theta)
        } else {
            ([0.0, 0.0, 0.0], 0.0)
        }
    } else {
        let s = 2.0 * theta.sin();
        (
            [
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s,
            ],
            theta,
        )
    };
    let w = scale3(axis, angle);
    let v = if angle < 1e-10 {
        t.translation
    } else {
        // V⁻¹ t = t − (θ/2)(k×t) + (1 − θ(1+cosθ)/(2 sinθ))(k×(k×t)).
        let sin_t = angle.sin();
        let c = 1.0 - angle * (1.0 + angle.cos()) / (2.0 * sin_t);
        let kt = cross3(axis, t.translation);
        let k2t = cross3(axis, kt);
        [
            t.translation[0] - 0.5 * angle * kt[0] + c * k2t[0],
            t.translation[1] - 0.5 * angle * kt[1] + c * k2t[1],
            t.translation[2] - 0.5 * angle * kt[2] + c * k2t[2],
        ]
    };
    [v[0], v[1], v[2], w[0], w[1], w[2]]
}

fn rodrigues(axis: &[f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let n = norm3(axis).max(1e-30);
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = angle.sin_cos();
    let cc = 1.0 - c;
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    [
        [
            c + cc * kx * kx,
            cc * kx * ky - s * kz,
            cc * kx * kz + s * ky,
        ],
        [
            cc * kx * ky + s * kz,
            c + cc * ky * ky,
            cc * ky * kz - s * kx,
        ],
        [
            cc * kx * kz - s * ky,
            cc * ky * kz + s * kx,
            c + cc * kz * kz,
        ],
    ]
}

fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn mul_mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm6(v: [f64; 6]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt()
}

fn dot6(a: [f64; 6], b: [f64; 6]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaw_pose(x: f64, y: f64, yaw: f64) -> Pose {
        let (s, c) = yaw.sin_cos();
        Pose {
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [x, y, 0.0],
        }
    }

    #[test]
    fn se3_log_exp_roundtrip() {
        let p = yaw_pose(1.0, -2.0, 0.7);
        let log = se3_log(&p);
        let v = [log[0], log[1], log[2]];
        let w = [log[3], log[4], log[5]];
        let identity = Pose::identity();
        let reconstructed = identity.left_perturb(&v, &w);
        for r in 0..3 {
            for c in 0..3 {
                assert!((reconstructed.rotation[r][c] - p.rotation[r][c]).abs() < 1e-9);
            }
            assert!((reconstructed.translation[r] - p.translation[r]).abs() < 1e-9);
        }
    }

    #[test]
    fn near_pi_log_roundtrip() {
        // Rotation by ~π about a tilted axis.
        let axis = [0.2, 0.5, 1.0];
        let theta = core::f64::consts::PI - 1e-7;
        let rot = rodrigues(&axis, theta);
        let p = Pose {
            rotation: rot,
            translation: [0.1, -0.2, 0.3],
        };
        let log = se3_log(&p);
        let v = [log[0], log[1], log[2]];
        let w = [log[3], log[4], log[5]];
        let reconstructed = Pose::identity().left_perturb(&v, &w);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (reconstructed.rotation[r][c] - p.rotation[r][c]).abs() < 1e-6,
                    "R[{r}][{c}]"
                );
            }
        }
    }

    #[test]
    fn cycle_graph_converges() {
        // Ground truth: a square trajectory of 4 poses with exact relative
        // edges, but node poses perturbed. Optimisation must reduce the
        // total cost.
        let truth = [
            yaw_pose(0.0, 0.0, 0.0),
            yaw_pose(2.0, 0.0, 0.0),
            yaw_pose(2.0, 2.0, core::f64::consts::FRAC_PI_2),
            yaw_pose(0.0, 2.0, core::f64::consts::PI),
        ];
        let mut graph = PoseGraph::new();
        let mut node_ids = Vec::new();
        for (i, t) in truth.iter().enumerate() {
            let jitter = yaw_pose(
                t.translation[0] + 0.05 * (i as f64).sin(),
                t.translation[1] - 0.04 * (i as f64).cos(),
                0.01 * (i as f64),
            );
            if i == 0 {
                node_ids.push(graph.add_node(*t)); // anchor fixed
            } else {
                node_ids.push(graph.add_node(jitter));
            }
        }
        for i in 0..4 {
            let j = (i + 1) % 4;
            let relative = truth[i].inverse().compose(&truth[j]);
            graph
                .add_edge(PoseEdge {
                    from: node_ids[i],
                    to: node_ids[j],
                    relative,
                    information: 1.0,
                    huber_delta: 0.0,
                })
                .unwrap();
        }
        let cost_before = graph.cost();
        let iterations = graph.optimize(&PoseGraphParams::default()).unwrap();
        let cost_after = graph.cost();
        assert!(
            cost_after < cost_before * 1e-3,
            "cost {cost_before} → {cost_after}"
        );
        assert!(iterations > 0);
        // Node 0 fixed.
        assert_eq!(graph.pose(0), Some(&truth[0]));
    }

    #[test]
    fn loop_closure_drift_is_corrected() {
        // Chain of 6 odometry edges with accumulated drift plus one loop
        // closure edge pinning the end back near the start.
        let mut graph = PoseGraph::new();
        graph.add_node(Pose::identity());
        let mut truth = vec![Pose::identity()];
        for i in 0..5 {
            let step = yaw_pose(0.5, 0.02 * (i as f64), 0.02);
            let last = *truth.last().unwrap();
            truth.push(last.compose(&step));
            let drifted = yaw_pose(step.translation[0] + 0.02, step.translation[1], 0.0);
            graph.add_node(last.compose(&drifted));
            graph
                .add_edge(PoseEdge {
                    from: i,
                    to: i + 1,
                    relative: step,
                    information: 1.0,
                    huber_delta: 0.0,
                })
                .unwrap();
        }
        // Loop closure: node 5 ≈ node 0 (the vehicle returned home).
        let loop_rel = truth[5].inverse().compose(&truth[0]);
        graph
            .add_edge(PoseEdge {
                from: 5,
                to: 0,
                relative: loop_rel,
                information: 1.0,
                huber_delta: 0.0,
            })
            .unwrap();
        let before = graph.cost();
        graph
            .optimize(&PoseGraphParams {
                max_iterations: 100,
                ..Default::default()
            })
            .unwrap();
        let after = graph.cost();
        assert!(after < before * 0.1, "cost {before} → {after}");
    }

    #[test]
    fn invalid_edges_rejected() {
        let mut graph = PoseGraph::new();
        graph.add_node(Pose::identity());
        assert!(graph
            .add_edge(PoseEdge {
                from: 0,
                to: 3,
                relative: Pose::identity(),
                information: 1.0,
                huber_delta: 0.0,
            })
            .is_err());
    }
}
