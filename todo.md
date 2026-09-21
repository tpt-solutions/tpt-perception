# tpt-perception — Phased Build Tracker

> Pure-Rust, AI-native 3D spatial perception & sensor fusion library. Dual-licensed **MIT OR Apache-2.0**. © TPT Solutions. See [spec.txt](spec.txt) for the full design.

## Open Decisions

- **Checkpoint:** `tpt-systems-optimisation` (used by `tpt-percept-slam` for pose graph optimization) is assumed to live under the same GitHub org as the rest of the `tpt-*` family, repo `tpt-systems-optimisation`. Confirm the exact URL before adding it as a git dependency in any `Cargo.toml`.
- **Checkpoint:** `tpt-gis` (used by `tpt-percept-map`/SLAM for WGS84/UTM transforms) is assumed to live under the same GitHub org, repo `tpt-gis`. Confirm the exact URL before adding it as a git dependency.
- **Resolved:** `tpt-control` does not exist yet (only `spec.txt`/`todo.md`, no code). `tpt-percept-fusion` will implement a self-contained EKF/UKF for now, with a later Checkpoint to swap in `tpt-control` once it's built.

## Per-Crate Checklist Template

Applied to every crate below (referenced by name, not repeated per crate):

1. Scaffold crate under `crates/`, add to workspace `Cargo.toml`.
2. Add only required dependencies (internal `tpt-*` crates or approved MIT/Apache deps — check `deny.toml`).
3. Implement the feature(s) listed for the crate.
4. Unit tests for the feature(s).
5. `rustdoc` on all public items.
6. `cargo fmt` + `cargo clippy --all-targets -- -D warnings` clean.
7. `cargo deny check` clean (no Apache-2.0-only deps).
8. `no_std` + `alloc` compatibility verified (where applicable per spec).

---

## Phase 0 — Foundation & Verification Setup (Weeks 1-2)

### Workspace scaffolding
- [ ] `git init`
- [ ] Root workspace `Cargo.toml` (members list, shared `[workspace.package]` with `license = "MIT OR Apache-2.0"`)
- [ ] `deny.toml` — allow-list MIT / Apache-2.0 / dual-licensed deps, deny Apache-2.0-only crates
- [ ] `LICENSE-MIT`, `LICENSE-APACHE` (copyright TPT Solutions)
- [ ] `rustfmt.toml`, clippy lint config
- [ ] `README.md` (project overview, link to spec.txt)
- [ ] `CONTRIBUTING.md`

### tpt-percept-core
- [ ] Phantom-typed coordinate frames (`Point3D<WorldFrame>`, `Point3D<SensorFrame>`, compile-time frame-mismatch prevention)
- [ ] Rigid body transforms: `Isometry3`, `Twist3` with const-generic dimensionality
- [ ] Unit-safe physical quantities (meters, radians, seconds) via `tpt-math-units` (path dep on sibling `tpt-math`)
- [ ] `no_std` + `alloc` compatibility
- [ ] Per-Crate Checklist Template

### tpt-percept-verify (initial)
- [ ] proptest strategies for points, transforms, noise profiles
- [ ] Kani harnesses for core geometric operations (frame transforms, distance preservation)
- [ ] Per-Crate Checklist Template

---

## Phase 1 — Point Cloud Processing (Weeks 3-5)

### tpt-percept-cloud
- [ ] Voxel grid hashing (sparse voxelization) for memory-efficient large-scale clouds
- [ ] Statistical outlier removal
- [ ] Radius filtering
- [ ] Pass-through filtering
- [ ] Spatial indexing: k-d tree
- [ ] Spatial indexing: octree
- [ ] Spatial indexing: R-tree
- [ ] Streaming point cloud processing for real-time LiDAR data
- [ ] Per-Crate Checklist Template

### tpt-percept-features
- [ ] Local surface normals, curvature, PCA
- [ ] Harris corner detection (3D)
- [ ] Edge detection
- [ ] Plane segmentation (RANSAC)
- [ ] FPFH (Fast Point Feature Histograms) descriptor
- [ ] SHOT descriptor
- [ ] Custom learned descriptor hooks (AI-native, autodiff-ready)
- [ ] Per-Crate Checklist Template

### Integration test
- [ ] Plane detection on synthetic point clouds, verified via proptest

---

## Phase 2 — Registration & SLAM (Weeks 6-8)

### tpt-percept-register
- [ ] ICP — point-to-point variant, with convergence guarantees
- [ ] ICP — point-to-plane variant, with convergence guarantees
- [ ] NDT (Normal Distributions Transform) probabilistic registration
- [ ] Feature-based registration (descriptor matching + RANSAC)
- [ ] Global registration: branch-and-bound
- [ ] Global registration: 3D Hough transform (loop closure support)
- [ ] Per-Crate Checklist Template

### tpt-percept-slam
- [ ] Visual odometry: feature tracking, pose estimation (monocular)
- [ ] Visual odometry: feature tracking, pose estimation (stereo)
- [ ] LiDAR odometry: scan matching, incremental map building
- [ ] Loop closure detection (place recognition, hand-crafted descriptors)
- [ ] Loop closure detection: learned descriptor hook (AI-native)
- [ ] **Checkpoint:** Pose graph optimization (sparse nonlinear least-squares) via `tpt-systems-optimisation` — confirm repo URL, add as git dependency
- [ ] Per-Crate Checklist Template

### tpt-percept-fusion
- [ ] Temporal alignment: timestamp interpolation, latency compensation
- [ ] Self-contained EKF/UKF integration for LiDAR+camera+IMU+radar — **Checkpoint:** swap to `tpt-control` once that crate exists
- [ ] Extrinsic calibration: automatic sensor-to-sensor transform estimation
- [ ] Uncertainty propagation: covariance intersection
- [ ] Robust fusion under sensor failure
- [ ] Per-Crate Checklist Template

### Verification
- [ ] Kani proof: ICP convergence bounds
- [ ] Kani proof: voxel grid indexing safety
- [ ] Kani proof / test: SLAM consistency under pose graph updates

---

## Phase 3 — Spatial Mapping (Weeks 9-10)

### tpt-percept-map
- [ ] Occupancy grids: 2D probabilistic maps (log-odds representation)
- [ ] Occupancy grids: 3D probabilistic maps (log-odds representation)
- [ ] Voxel maps: TSDF (Truncated Signed Distance Function) surface reconstruction
- [ ] Signed distance fields for collision checking and trajectory optimization
- [ ] Semantic maps: labeled point clouds for object recognition / scene understanding
- [ ] **Checkpoint:** WGS84/UTM coordinate transforms via `tpt-gis` for outdoor SLAM and multi-robot coordination — confirm repo URL, add as git dependency
- [ ] Per-Crate Checklist Template

---

## Phase 4 — Umbrella, Docs & Release v1.0 (Weeks 11-12)

### tpt-perception (umbrella crate)
- [ ] Feature-gated re-exports of all 8 sub-crates
- [ ] Crate-level documentation and usage examples
- [ ] Per-Crate Checklist Template

### tpt-percept-verify (hardening)
- [ ] Full Kani proof suite wired into CI across all crates
- [ ] proptest coverage pass across all geometric invariants (rigid transforms preserve distances, ICP residual monotonically decreases, etc.)

### Release readiness
- [ ] `cargo deny check` clean across full workspace
- [ ] Docs build clean (`cargo doc --no-deps`)
- [ ] Version tag `v1.0`

---

## Integration & Synergies (tracked, not phase-gated)

- [ ] `tpt-dsp`: FFT / filtering primitives for `tpt-percept-fusion` (sensor pre-filtering) and `tpt-percept-slam` (frequency-domain feature extraction)
- [ ] AI-native agent hooks: expose autodiff on feature descriptors and cost functions for `tpt-eve` / `tpt-anima` learned perception or RL-based navigation
