# tpt-percept-slam

SLAM for `tpt-perception`: odometry, place recognition, and pose graph
optimization.

- **Monocular visual odometry** — normalized 8-point essential matrix
  (9×9 cyclic-Jacobi smallest eigenvector), singular-value projection onto
  the essential manifold, four-fold decomposition with exact-depth
  cheirality selection, RANSAC over correspondences. Motion up to scale.
- **Stereo visual odometry** — disparity triangulation (`Z = f·b/d`) and
  metric 3-D-3-D RANSAC/Kabsch alignment.
- **LiDAR odometry** — point-to-plane scan matching with constant-velocity
  prediction, poses accumulated in the odometry frame.
- **Loop closure** — Scan Context descriptors (polar grid, ring-key
  screening, azimuth-shift matching).
- **Pose graph** — dense Gauss–Newton over SE(3) with numeric Jacobians,
  Huber robustification, fixed anchor node. Intentionally small-scale; the
  sparse large-scale solver is the open `tpt-systems-optimisation`
  checkpoint in `todo.md`.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.
