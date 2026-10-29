//! Pose graph optimization: dense Levenberg–Marquardt over SE(3) poses with
//! numeric Jacobians.
//!
//! Nodes carry poses; edges carry *relative* pose measurements
//! `T_ij ≈ T_i⁻¹ · T_j` (odometry, loop closure). The optimiser minimises
//! `Σ ‖log(T_ij_meas⁻¹ · T_i⁻¹ T_j)‖²_{Ω}` — the SE(3) log residual with an
//! optional Huber robustifier — by damped Gauss–Newton with left-multiplied
//! perturbations. Edges take a scalar weight or a full 6×6 information
//! matrix.
//!
//! **Scale contract:** the solver is intentionally dense (O(n²) memory,
//! numeric Jacobians) and targets graphs of up to a few hundred nodes —
//! enough for loop-closure batches. Large-scale sparse optimization is the
//! open `tpt-systems-optimisation` checkpoint in `todo.md`; swap the solver
//! there when that crate becomes available.
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg_dense::DVector;
use tpt_math_linalg_sparse::{CooMatrix, SparseCholesky};
#[cfg_attr(feature = "std", allow(unused_imports))]
use tpt_math_numeric::Float as _;

use crate::error::SlamError;

/// A pose as `(R, t)` — `p_world = R · p_body + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// SE(3) exponential of a tangent vector `ξ = (v, ω)`: the pose with
    /// `exp(ξ) = Pose::identity().left_perturb(v, ω)`.
    pub fn exp(xi: &[f64; 6]) -> Pose {
        Pose::identity().left_perturb(&[xi[0], xi[1], xi[2]], &[xi[3], xi[4], xi[5]])
    }

    /// SE(3) logarithm: the tangent vector `ξ = (v, ω)` with
    /// `Pose::exp(&ξ) == self`, up to numerical precision.
    pub fn log(&self) -> [f64; 6] {
        se3_log(self)
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoseEdge {
    /// Source node.
    pub from: usize,
    /// Target node.
    pub to: usize,
    /// Measured relative pose `T_i⁻¹ · T_j`.
    pub relative: Pose,
    /// Information scale (weight). The effective information is
    /// `information · Ω`, where `Ω` is [`Self::information_matrix`] or the
    /// identity when that is `None`.
    pub information: f64,
    /// Optional full 6×6 information matrix `Ω` over the tangent ordering
    /// `(v, ω)`. Must be symmetric; typically the inverse of the
    /// registration covariance.
    pub information_matrix: Option<[[f64; 6]; 6]>,
    /// Huber robustifier delta (0 disables robustification).
    pub huber_delta: f64,
}

/// Outcome of [`PoseGraph::optimize`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizeReport {
    /// Outer iterations executed.
    pub iterations: u32,
    /// True when the cost reached the tolerance or was (numerically) zero.
    /// False means the iteration cap was hit or the solver stalled.
    pub converged: bool,
    /// Cost before optimisation.
    pub initial_cost: f64,
    /// Cost after optimisation.
    pub final_cost: f64,
}

/// Optimizer settings.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoseGraphParams {
    /// Maximum Gauss–Newton iterations.
    pub max_iterations: u32,
    /// Stop when the total cost decreases by less than this (relative).
    pub tolerance: f64,
    /// Forward-difference step for numeric Jacobians.
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
        if let Some(omega) = edge.information_matrix {
            let finite = omega.iter().flatten().all(|v| v.is_finite());
            let symmetric = (0..6).all(|r| (0..r).all(|c| omega[r][c] == omega[c][r]));
            if !finite || !symmetric {
                return Err(SlamError::InvalidParameter(
                    "information_matrix must be finite and symmetric",
                ));
            }
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

    /// Total Huber-weighted, information-weighted squared constraint cost.
    pub fn cost(&self) -> f64 {
        total_cost(&self.edges, &self.poses)
    }

    /// Runs Levenberg–Marquardt Gauss–Newton over the free nodes (node 0 is
    /// held fixed).
    ///
    /// The normal equations are assembled as a sparse matrix and solved with a
    /// sparse Cholesky factorisation, so cost scales with the graph's sparsity
    /// rather than with `(6N)²`. A step is accepted only when it lowers the
    /// cost; otherwise the damping is raised and the step recomputed. The graph
    /// is never cloned: only the pose vector is copied per trial step.
    ///
    /// Returns an [`OptimizeReport`]. Failure to converge (iteration cap, or
    /// no descent at any damping level) is reported through
    /// [`OptimizeReport::converged`] rather than as an error.
    pub fn optimize(&mut self, params: &PoseGraphParams) -> Result<OptimizeReport, SlamError> {
        let initial_cost = self.cost();
        let mut report = OptimizeReport {
            iterations: 0,
            converged: false,
            initial_cost,
            final_cost: initial_cost,
        };
        let n = self.poses.len();
        if n < 2 || self.edges.is_empty() {
            report.converged = true;
            return Ok(report);
        }
        let dim = 6 * (n - 1);
        // Bandwidth-reducing order of the free nodes: keeps Cholesky fill low,
        // especially for loop closures that would otherwise couple the two
        // ends of a long chain.
        let pos = reverse_cuthill_mckee(n, &self.edges);
        let mut cost = initial_cost;
        let mut lambda = LM_INITIAL_DAMPING;

        for _ in 0..params.max_iterations {
            // Stop once the cost is negligible relative to where we started.
            // Scaling by the initial cost keeps the test meaningful when the
            // residual is already at round-off level.
            let floor = (params.tolerance * initial_cost).max(ABSOLUTE_COST_FLOOR);
            if cost <= floor {
                report.converged = true;
                break;
            }
            report.iterations += 1;
            let (h, rhs) = self.normal_equations(params.numeric_step, dim, &pos);
            let rhs = DVector::from_row_slice(&rhs);
            let diag = diagonal_of(&h);

            let mut accepted: Option<(Vec<Pose>, f64, f64)> = None;
            for _ in 0..LM_MAX_TRIES {
                let damped = damped_matrix(&h, &diag, lambda);
                let solved = SparseCholesky::factor(&damped).and_then(|f| f.solve(&rhs));
                let Ok(delta) = solved else {
                    lambda *= LM_UP;
                    continue;
                };

                let mut candidate = self.poses.clone();
                let mut max_step = 0.0f64;
                for node in 1..n {
                    let base = pos[node] * 6;
                    let v = [delta[base], delta[base + 1], delta[base + 2]];
                    let w = [delta[base + 3], delta[base + 4], delta[base + 5]];
                    candidate[node] = candidate[node].left_perturb(&v, &w);
                    max_step = max_step.max(norm3(&v)).max(norm3(&w));
                }
                let new_cost = total_cost(&self.edges, &candidate);
                if new_cost < cost {
                    accepted = Some((candidate, new_cost, max_step));
                    break;
                }
                lambda *= LM_UP;
            }

            let Some((poses, new_cost, max_step)) = accepted else {
                break;
            };
            let relative_decrease = (cost - new_cost) / cost;
            self.poses = poses;
            cost = new_cost;
            lambda = (lambda * LM_DOWN).max(LM_MIN);

            if relative_decrease < params.tolerance || max_step < 1e-12 {
                report.converged = true;
                break;
            }
        }

        report.final_cost = cost;
        Ok(report)
    }

    /// Accumulates the normal equations `H δ = rhs` over all edges as sparse
    /// triplets, with node 0 removed. `rhs = −Σ Jᵀ Ω r` (Gauss–Newton descent).
    /// Duplicate triplets are summed when the matrix is compressed.
    fn normal_equations(&self, step: f64, dim: usize, pos: &[usize]) -> (CooMatrix<f64>, Vec<f64>) {
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        let mut rhs = vec![0.0f64; dim];

        for edge in &self.edges {
            let pi = self.poses[edge.from];
            let pj = self.poses[edge.to];
            let r0 = residual_with(edge, &pi, &pj);

            // Forward-difference Jacobian of the residual w.r.t. the left
            // perturbation of both endpoints. Columns 0..6 belong to `from`,
            // 6..12 to `to`.
            let mut jac = [[0.0f64; 12]; 6];
            for k in 0..6 {
                let mut xi = [0.0f64; 6];
                xi[k] = step;
                let (v, w) = ([xi[0], xi[1], xi[2]], [xi[3], xi[4], xi[5]]);
                let rp = residual_with(edge, &pi.left_perturb(&v, &w), &pj);
                let rq = residual_with(edge, &pi, &pj.left_perturb(&v, &w));
                for row in 0..6 {
                    jac[row][k] = (rp[row] - r0[row]) / step;
                    jac[row][6 + k] = (rq[row] - r0[row]) / step;
                }
            }

            let weight = edge.information_block_scaled(&r0);
            let wr = mat6_vec(&weight, r0);
            let wj = mat6_times_jac(&weight, &jac);

            let global = |col: usize| -> Option<usize> {
                let node = if col < 6 { edge.from } else { edge.to };
                if node == 0 {
                    None
                } else {
                    Some(pos[node] * 6 + (col % 6))
                }
            };
            for a in 0..12 {
                let Some(ga) = global(a) else { continue };
                for b in 0..12 {
                    let Some(gb) = global(b) else { continue };
                    let mut acc = 0.0;
                    for r in 0..6 {
                        acc += jac[r][a] * wj[r][b];
                    }
                    rows.push(ga);
                    cols.push(gb);
                    vals.push(acc);
                }
                let mut acc = 0.0;
                for r in 0..6 {
                    acc += jac[r][a] * wr[r];
                }
                rhs[ga] -= acc;
            }
        }
        (CooMatrix::from_triplets(dim, dim, rows, cols, vals), rhs)
    }
}

/// Reverse Cuthill–McKee ordering of the free nodes (node 0 is the fixed
/// anchor and is excluded). Returns `pos`, where `pos[node]` is the node's
/// rank in the ordering, so node `k` occupies rows `6·pos[k] … 6·pos[k] + 6`.
/// Entry 0 of `pos` is unused.
fn reverse_cuthill_mckee(n: usize, edges: &[PoseEdge]) -> Vec<usize> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in edges {
        if e.from != 0 && e.to != 0 && e.from != e.to {
            adj[e.from].push(e.to);
            adj[e.to].push(e.from);
        }
    }
    let degree: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    let mut visited = vec![false; n];
    visited[0] = true;
    let mut order: Vec<usize> = Vec::with_capacity(n.saturating_sub(1));
    // Start each connected component from its lowest-degree unvisited node.
    let mut candidates: Vec<usize> = (1..n).collect();
    candidates.sort_by_key(|&i| degree[i]);
    for start in candidates {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = alloc::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            let mut next: Vec<usize> = adj[u].iter().copied().filter(|&v| !visited[v]).collect();
            next.sort_by_key(|&v| degree[v]);
            next.dedup();
            for v in next {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }
    order.reverse();
    let mut pos = vec![0usize; n];
    for (rank, &node) in order.iter().enumerate() {
        pos[node] = rank;
    }
    pos
}

/// Diagonal of a sparse matrix (duplicates already summed by `to_csr`).
fn diagonal_of(h: &CooMatrix<f64>) -> Vec<f64> {
    let csr = h.to_csr();
    let mut diag = vec![0.0f64; csr.nrows()];
    for (r, c, v) in csr.iter() {
        if r == c {
            diag[r] = *v;
        }
    }
    diag
}

/// `H + λ · max(diag(H), floor)` as a compressed SPD matrix.
fn damped_matrix(
    h: &CooMatrix<f64>,
    diag: &[f64],
    lambda: f64,
) -> tpt_math_linalg_sparse::CsrMatrix<f64> {
    let base = h.to_csr();
    let n = diag.len();
    let mut rows: Vec<usize> = Vec::with_capacity(base.nnz() + n);
    let mut cols: Vec<usize> = Vec::with_capacity(base.nnz() + n);
    let mut vals: Vec<f64> = Vec::with_capacity(base.nnz() + n);
    for (r, c, v) in base.iter() {
        rows.push(r);
        cols.push(c);
        vals.push(*v);
    }
    for (i, d) in diag.iter().enumerate() {
        rows.push(i);
        cols.push(i);
        vals.push(lambda * d.max(DIAG_FLOOR));
    }
    CooMatrix::from_triplets(n, n, rows, cols, vals).to_csr()
}

impl PoseEdge {
    /// Effective 6×6 information `information · Ω` (Ω = `information_matrix`
    /// or the identity), before any robust reweighting.
    fn information_block(&self) -> [[f64; 6]; 6] {
        let mut m = self.information_matrix.unwrap_or(IDENTITY6);
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v *= self.information;
            }
        }
        m
    }

    /// [`Self::information_block`] further scaled by the Huber weight for
    /// the residual `r`.
    fn information_block_scaled(&self, r: &[f64; 6]) -> [[f64; 6]; 6] {
        let mut m = self.information_block();
        let scale = self.huber_scale(r);
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        m
    }

    /// Huber weight `δ / ‖r‖` beyond `δ` (1 inside, or when disabled).
    fn huber_scale(&self, r: &[f64; 6]) -> f64 {
        if self.huber_delta > 0.0 {
            let n = norm6(*r);
            if n > self.huber_delta {
                return self.huber_delta / n;
            }
        }
        1.0
    }
}

const IDENTITY6: [[f64; 6]; 6] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

/// Initial Levenberg–Marquardt damping (relative to the Hessian diagonal).
const LM_INITIAL_DAMPING: f64 = 1e-4;
/// Damping multiplier after a rejected step.
const LM_UP: f64 = 10.0;
/// Damping multiplier after an accepted step.
const LM_DOWN: f64 = 0.1;
/// Lower bound on damping.
const LM_MIN: f64 = 1e-12;
/// Damping retries per iteration before giving up on the iteration.
const LM_MAX_TRIES: usize = 12;
/// Diagonal floor so isolated nodes still yield a solvable system.
const DIAG_FLOOR: f64 = 1e-9;
/// Cost below which the graph is treated as exactly consistent.
const ABSOLUTE_COST_FLOOR: f64 = 1e-20;

/// Sum of edge costs `Σ s·rᵀ Ω r` at the given node poses.
fn total_cost(edges: &[PoseEdge], poses: &[Pose]) -> f64 {
    edges
        .iter()
        .map(|e| {
            let r = residual_with(e, &poses[e.from], &poses[e.to]);
            e.huber_scale(&r) * quad6(&e.information_block(), r)
        })
        .sum()
}

/// SE(3) log residual `log(T_ij_meas⁻¹ · T_i⁻¹ · T_j)` of an edge for the
/// given endpoint poses.
fn residual_with(edge: &PoseEdge, ti: &Pose, tj: &Pose) -> [f64; 6] {
    let measured_inv = edge.relative.inverse();
    let actual = ti.inverse().compose(tj);
    se3_log(&measured_inv.compose(&actual))
}

fn mat6_vec(m: &[[f64; 6]; 6], v: [f64; 6]) -> [f64; 6] {
    let mut out = [0.0f64; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r] += m[r][c] * v[c];
        }
    }
    out
}

fn mat6_times_jac(m: &[[f64; 6]; 6], j: &[[f64; 12]; 6]) -> [[f64; 12]; 6] {
    let mut out = [[0.0f64; 12]; 6];
    for r in 0..6 {
        for c in 0..12 {
            for k in 0..6 {
                out[r][c] += m[r][k] * j[k][c];
            }
        }
    }
    out
}

fn quad6(m: &[[f64; 6]; 6], r: [f64; 6]) -> f64 {
    dot6(r, mat6_vec(m, r))
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
                    information_matrix: None,
                    huber_delta: 0.0,
                })
                .unwrap();
        }
        let cost_before = graph.cost();
        let report = graph.optimize(&PoseGraphParams::default()).unwrap();
        let cost_after = graph.cost();
        assert!(
            cost_after < cost_before * 1e-3,
            "cost {cost_before} → {cost_after}"
        );
        assert!(report.iterations > 0);
        assert!(report.converged);
        assert_eq!(report.final_cost, cost_after);
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
                    information_matrix: None,
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
                information_matrix: None,
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

    fn translation_pose(x: f64) -> Pose {
        Pose {
            translation: [x, 0.0, 0.0],
            ..Pose::identity()
        }
    }

    #[test]
    fn matrix_information_matches_equivalent_scalar_weight() {
        // Two conflicting measurements of node 1 from the fixed anchor:
        // x = 1.0 with weight 100, x = 2.0 with weight 1. The optimum is the
        // weighted mean (100·1 + 1·2) / 101.
        let expected = (100.0 + 2.0) / 101.0;

        let solve_with = |heavy: PoseEdge, light: PoseEdge| {
            let mut g = PoseGraph::new();
            g.add_node(Pose::identity());
            g.add_node(translation_pose(0.0));
            g.add_edge(heavy).unwrap();
            g.add_edge(light).unwrap();
            g.optimize(&PoseGraphParams::default()).unwrap();
            g.pose(1).unwrap().translation[0]
        };

        let scalar = solve_with(
            PoseEdge {
                from: 0,
                to: 1,
                relative: translation_pose(1.0),
                information: 100.0,
                information_matrix: None,
                huber_delta: 0.0,
            },
            PoseEdge {
                from: 0,
                to: 1,
                relative: translation_pose(2.0),
                information: 1.0,
                information_matrix: None,
                huber_delta: 0.0,
            },
        );
        assert!((scalar - expected).abs() < 1e-6, "scalar {scalar}");

        // Same weighting expressed as a full matrix, with the heavy edge's
        // off-diagonal coupling ignored by a diagonal Ω.
        let mut omega = [[0.0f64; 6]; 6];
        for (i, row) in omega.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let mut heavy_omega = omega;
        heavy_omega[0][0] = 100.0;
        let matrix = solve_with(
            PoseEdge {
                from: 0,
                to: 1,
                relative: translation_pose(1.0),
                information: 1.0,
                information_matrix: Some(heavy_omega),
                huber_delta: 0.0,
            },
            PoseEdge {
                from: 0,
                to: 1,
                relative: translation_pose(2.0),
                information: 1.0,
                information_matrix: Some(omega),
                huber_delta: 0.0,
            },
        );
        assert!((matrix - expected).abs() < 1e-6, "matrix {matrix}");
    }

    #[test]
    fn isolated_node_still_solves_and_reports() {
        // Node 2 has no edges: its columns are zero, damping keeps the
        // system solvable and the connected part still converges.
        let mut g = PoseGraph::new();
        g.add_node(Pose::identity());
        g.add_node(translation_pose(0.5));
        g.add_node(translation_pose(3.0));
        g.add_edge(PoseEdge {
            from: 0,
            to: 1,
            relative: translation_pose(1.0),
            information: 1.0,
            information_matrix: None,
            huber_delta: 0.0,
        })
        .unwrap();
        let report = g.optimize(&PoseGraphParams::default()).unwrap();
        assert!(report.final_cost <= report.initial_cost);
        assert!(report.converged);
        assert!((g.pose(1).unwrap().translation[0] - 1.0).abs() < 1e-6);
        assert_eq!(g.pose(2).unwrap().translation[0], 3.0);
    }

    #[test]
    fn iteration_cap_reports_not_converged() {
        let mut g = PoseGraph::new();
        g.add_node(Pose::identity());
        g.add_node(translation_pose(0.5));
        g.add_edge(PoseEdge {
            from: 0,
            to: 1,
            relative: translation_pose(1.0),
            information: 1.0,
            information_matrix: None,
            huber_delta: 0.0,
        })
        .unwrap();
        let report = g
            .optimize(&PoseGraphParams {
                max_iterations: 0,
                ..Default::default()
            })
            .unwrap();
        assert!(!report.converged);
        assert_eq!(report.iterations, 0);
        assert_eq!(report.final_cost, report.initial_cost);
    }

    #[test]
    fn large_ring_graph_converges_with_sparse_solver() {
        // 150 poses on a circle, odometry edges between neighbours, a loop
        // closure back to the start. Initial guesses drift by a small bias
        // per step, so the optimiser has real work to do.
        const N: usize = 150;
        let point = |i: usize| {
            let a = 2.0 * core::f64::consts::PI * i as f64 / N as f64;
            Pose {
                translation: [10.0 * a.cos(), 10.0 * a.sin(), 0.0],
                ..Pose::identity()
            }
        };
        let truth: Vec<Pose> = (0..=N).map(|i| point(i % N)).collect();

        let mut graph = PoseGraph::new();
        let mut estimate = Pose::identity();
        graph.add_node(estimate);
        for i in 0..N {
            let relative = truth[i].inverse().compose(&truth[i + 1]);
            // Drifting initial guess: the chain accumulates a bias.
            let biased = Pose {
                translation: [
                    relative.translation[0] + 0.002,
                    relative.translation[1],
                    relative.translation[2],
                ],
                ..relative
            };
            estimate = estimate.compose(&biased);
            graph.add_node(estimate);
            graph
                .add_edge(PoseEdge {
                    from: i,
                    to: i + 1,
                    relative,
                    information: 1.0,
                    information_matrix: None,
                    huber_delta: 0.0,
                })
                .unwrap();
        }
        // Loop closure: last node is the first.
        graph
            .add_edge(PoseEdge {
                from: N,
                to: 0,
                relative: Pose::identity(),
                information: 100.0,
                information_matrix: None,
                huber_delta: 0.0,
            })
            .unwrap();

        let report = graph
            .optimize(&PoseGraphParams {
                max_iterations: 100,
                ..Default::default()
            })
            .unwrap();
        assert!(report.converged, "{report:?}");
        assert!(
            report.final_cost < report.initial_cost * 1e-3,
            "cost {} → {}",
            report.initial_cost,
            report.final_cost
        );
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
                information_matrix: None,
                huber_delta: 0.0,
            })
            .is_err());
    }
}
