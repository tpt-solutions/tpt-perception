# tpt-percept-cloud

Point cloud data structures and processing for `tpt-perception`.

- **Containers** — `PointCloud` (raw) and `FrameCloud<F>` (frame-tagged so
  clouds from different sensors cannot be mixed without an explicit
  `Isometry3<F1, F2>`).
- **Voxel grids** — sparse lattice hashing with saturating, panic-free key
  computation; voxel-grid downsampling by centroid.
- **Filters** — statistical outlier removal (kNN mean-distance threshold),
  radius outlier removal, pass-through cropping. All return survivor
  indices so per-point attributes filter alongside.
- **Spatial indexing** — balanced k-d tree (exact k-NN and radius queries,
  squared-distance API), region octree (ball and box queries), static
  R-tree (STR-style bulk load over AABBs).
- **Streaming** — sliding scan windows and composable filter stages for
  real-time LiDAR pipelines.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.
