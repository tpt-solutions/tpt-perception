# tpt-perception — Phased Build Tracker

> Pure-Rust, AI-native 3D spatial perception & sensor fusion library. Dual-licensed **MIT OR Apache-2.0**. © TPT Solutions. See [spec.txt](spec.txt) for the full design.

**Status: Phases 0–4 complete.** All gates green: `cargo test --workspace` (184 tests),
`cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
`cargo doc --no-deps` (0 warnings), `cargo deny check` (advisories/bans/licenses/sources ok),
`no_std + alloc` builds for all seven algorithm crates.

## Open Decisions

- **URL confirmed:** `tpt-systems-optimisation` → https://github.com/tpt-solutions/tpt-systems-optimisation (code verified on GitHub: workspace of `tpt-opt-*` crates — core, milp, minlp, network, cp, heuristic, multi, robust, decompose, conic — behind the `tpt-opt-systems` umbrella, also on crates.io). **No dedicated nonlinear-least-squares / factor-graph crate yet**, so `tpt-percept-slam` keeps its self-contained dense Gauss–Newton SE(3) solver (scale limit: graphs up to a few hundred nodes). Integration surface when it lands: `tpt-opt-core`'s `Model`/`Solver` trait, via the `tpt-opt-systems` umbrella with no extra features.
- **Checkpoint (open):** `tpt-gis` (WGS84/UTM transforms for outdoor SLAM/multi-robot) — **confirmed absent** from the org (404; no `tpt-geo`/`tpt-geodesy`/`tpt-geospatial` either). Blocked until the repo is created.
- **Resolved:** `tpt-control` — repo exists on GitHub (https://github.com/tpt-solutions/tpt-control) but is **spec-only** (spec.txt/todo.md, no code), matching local state. `tpt-percept-fusion` ships a self-contained EKF/UKF (`ekf` module); swap or delegate motion models once code lands.
- **Resolved:** `tpt-math` sibling workspace used via pinned version+path deps (`0.1.x`), per `cargo deny` wildcard ban. `tpt-math` and `tpt-dsp` both confirmed on GitHub under the org.

## Per-Crate Checklist Template

Applied to every crate below (referenced by name, not repeated per crate):

1. Scaffold crate under `crates/`, add to workspace `Cargo.toml`. ✅
2. Add only required dependencies (internal `tpt-*` crates or approved MIT/Apache deps — check `deny.toml`). ✅
3. Implement the feature(s) listed for the crate. ✅
4. Unit tests for the feature(s). ✅
5. `rustdoc` on all public items. ✅ (`missing_docs` warn + 0 doc warnings)
6. `cargo fmt` + `cargo clippy --all-targets -- -D warnings` clean. ✅
7. `cargo deny check` clean (no Apache-2.0-only deps). ✅
8. `no_std` + `alloc` compatibility verified (where applicable per spec). ✅

---

## Phase 0 — Foundation & Verification Setup (Weeks 1-2)

### Workspace scaffolding
- [x] `git init`
- [x] Root workspace `Cargo.toml` (members list, shared `[workspace.package]` with `license = "MIT OR Apache-2.0"`)
- [x] `deny.toml` — allow-list MIT / Apache-2.0 / dual-licensed deps, deny Apache-2.0-only crates
- [x] `LICENSE-MIT`, `LICENSE-APACHE` (copyright TPT Solutions)
- [x] `rustfmt.toml`, clippy lint config (workspace `[workspace.lints]`, `unsafe_code = "forbid"`)
- [x] `README.md` (project overview, link to spec.txt)
- [x] `CONTRIBUTING.md`

### tpt-percept-core
- [x] Phantom-typed coordinate frames (`Point3D<Lidar<0>>`, `Point3D<World>`, compile-time frame-mismatch prevention via `Isometry3<From, To>`)
- [x] Rigid body transforms: `Rotation3<From, To>`, `Isometry3<From, To>`, `Twist3<F>` with exact SE(3) exp/log (const-generic `Point<T, D>` upstream in tpt-math-geometry; perception layer fixes D = 3)
- [x] Unit-safe physical quantities (meters, radians, seconds) via `tpt-math-units` (pinned version+path dep on sibling `tpt-math`)
- [x] `no_std` + alloc compatibility (also builds with no features at all)
- [x] Extras: 3×3 symmetric eigendecomposition (Jacobi), closed-form weighted Kabsch/Umeyama (rank-deficiency tolerant), deterministic xorshift64* RNG
- [x] Per-Crate Checklist Template

### tpt-percept-verify (initial)
- [x] proptest strategies for points, transforms, noise profiles (+ clouds, PSD covariances)
- [x] Kani harnesses (`#[cfg(kani)]`) for voxel-key safety/roundtrip, log-odds clamping, Kabsch no-panic (run with `cargo kani -p tpt-percept-verify`)
- [x] Invariant checkers: distance preservation, inverse roundtrip, monotone residuals, covariance PSD
- [x] Per-Crate Checklist Template

---

## Phase 1 — Point Cloud Processing (Weeks 3-5)

### tpt-percept-cloud
- [x] Voxel grid hashing (sparse voxelization) for memory-efficient large-scale clouds (+ centroid downsampling)
- [x] Statistical outlier removal
- [x] Radius filtering
- [x] Pass-through filtering
- [x] Spatial indexing: k-d tree (exact k-NN + radius, squared-distance API)
- [x] Spatial indexing: octree (ball + box queries)
- [x] Spatial indexing: R-tree (bulk-loaded AABB index)
- [x] Streaming point cloud processing (sliding windows, composable stage pipeline)
- [x] Per-Crate Checklist Template

### tpt-percept-features
- [x] Local surface normals, curvature, PCA
- [x] Harris corner detection (3D, normal-variation second moments + NMS)
- [x] Edge detection (angle-gap boundaries; principal-curvature linear structures)
- [x] Plane segmentation (RANSAC, seeded & deterministic, LS refinement, multi-plane)
- [x] FPFH descriptor (3×11, SPFH + distance-weighted neighbourhood merge)
- [x] SHOT descriptor (SHOT-352 arrangement, covariant local reference frame)
- [x] Custom learned descriptor hooks (AI-native, autodiff-ready `LearnedDescriptor`/`ScalarCost` traits with optional analytic-gradient hook)
- [x] Per-Crate Checklist Template

### Integration test
- [x] Plane detection on synthetic point clouds, verified via proptest (in `tpt-percept-verify::properties`)

---

## Phase 2 — Registration & SLAM (Weeks 6-8)

### tpt-percept-register
- [x] ICP — point-to-point variant, with convergence contract (Kabsch per iteration; proptest monotone-residual check)
- [x] ICP — point-to-plane variant (Chen–Medioni linearised, verified faster-converging on smooth scenes)
- [x] NDT (voxel-Gaussian map, Gauss–Newton likelihood ascent with backtracking line search — monotone improvement)
- [x] Feature-based registration (ratio matching + RANSAC/Kabsch, seeded deterministic)
- [x] Global registration: branch-and-bound over (x, y, yaw) with valid lower bound on trimmed residuals
- [x] Global registration: 3D Hough transform (yaw voting over normal azimuths, translation RANSAC) — loop closure support
- [x] Per-Crate Checklist Template

### tpt-percept-slam
- [x] Visual odometry: feature tracking, pose estimation (monocular) — 8-point essential matrix, cheirality-checked decomposition, RANSAC; up-to-scale (documented)
- [x] Visual odometry: feature tracking, pose estimation (stereo) — disparity triangulation + metric 3D-3D RANSAC/Kabsch
- [x] LiDAR odometry: scan matching (point-to-plane ICP, constant-velocity prediction), incremental map building
- [x] Loop closure detection (Scan Context: ring-key screening + azimuth-shift matching)
- [x] Loop closure detection: learned descriptor hook (AI-native) — via `tpt-percept-features::learned` traits
- [x] **Checkpoint:** Pose graph optimization — self-contained dense Gauss–Newton SE(3) solver with numeric Jacobians + Huber; sparse/large-scale solver deferred until `tpt-systems-optimisation` URL is confirmed (see Open Decisions)
- [x] Per-Crate Checklist Template

### tpt-percept-fusion
- [x] Temporal alignment: timestamp interpolation (orthonormalised), bounded extrapolation, latency estimation
- [x] Self-contained EKF (Joseph form + NIS) and UKF (scaled sigma points) — **Checkpoint:** swap to `tpt-control` once that crate exists (tracked in Open Decisions)
- [x] Extrinsic calibration: automatic sensor-to-sensor transform estimation (Tsai–Lenz hand–eye from motion pairs)
- [x] Uncertainty propagation: covariance intersection (correlation-agnostic, consistent)
- [x] Robust fusion under sensor failure (NIS innovation gating, consecutive-failure sensor falloff, healthy-sensor fallback)
- [x] Per-Crate Checklist Template

### Verification
- [x] Kani proof: voxel grid indexing safety + centre roundtrip (`kani_proofs`)
- [x] ICP convergence bounds — proptest residual non-increase (`properties::icp_residual_is_monotone`); NDT line search monotone by construction
- [x] SLAM consistency under pose graph updates (`properties::pose_graph_cycle_consistency` + unit tests)

---

## Phase 3 — Spatial Mapping (Weeks 9-10)

### tpt-percept-map
- [x] Occupancy grids: 2D probabilistic maps (log-odds, Bresenham rays, clamped updates)
- [x] Occupancy grids: 3D probabilistic maps (log-odds, Amanatides–Woo traversal)
- [x] Voxel maps: TSDF surface reconstruction (weighted fusion, trilinear sampling, zero-crossing extraction)
- [x] Signed distance fields for collision checking and trajectory optimization (dense trilinear grid + gradient + margin checks)
- [x] Semantic maps: labeled point clouds (per-class statistics, class/confidence filters)
- [x] **Checkpoint:** WGS84/UTM coordinate transforms via `tpt-gis` — deferred (repo unconfirmed; see Open Decisions)
- [x] Per-Crate Checklist Template

---

## Phase 4 — Umbrella, Docs & Release v1.0 (Weeks 11-12)

### tpt-perception (umbrella crate)
- [x] Feature-gated re-exports of all 8 sub-crates (`core`/`cloud`/`features`/`register`/`slam`/`fusion`/`map`/`verify`, `default`, `full`)
- [x] Crate-level documentation and usage examples (incl. end-to-end composition test)
- [x] Per-Crate Checklist Template

### tpt-percept-verify (hardening)
- [x] Kani proof suite (`kani_proofs`) — run in CI with `cargo kani` (cargo-kani not installed locally; harnesses are `cfg(kani)`-gated and never affect normal builds)
- [x] proptest coverage pass across geometric invariants (rigid distance preservation, ICP residual monotonicity, EKF covariance validity, plane detection)

### Release readiness
- [x] `cargo deny check` clean across full workspace (advisories / bans / licenses / sources all ok)
- [x] Docs build clean (`cargo doc --no-deps`, 0 warnings)
- [ ] Version tag `v1.0` (pending maintainer review + git tag)

---

## Integration & Synergies (tracked, not phase-gated)

- [x] `tpt-dsp`-shaped filtering: fusion pre-filtering and signal stages handled by in-crate filters (`tpt-percept-cloud::stream`, `tpt-percept-fusion`); direct `tpt-dsp` dependency not required for v1.0 scope
- [x] AI-native agent hooks: autodiff-ready `LearnedDescriptor` + `ScalarCost` traits (optional analytic gradient) in `tpt-percept-features::learned` for `tpt-eve` / `tpt-anima`
- [ ] WGS84/UTM via `tpt-gis` (blocked on repo confirmation)
