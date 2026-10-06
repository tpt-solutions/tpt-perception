# tpt-percept-features

Geometric feature extraction for `tpt-perception`.

- **Normals & curvature** — local-PCA over kNN neighbourhoods; viewpoint
  orientation. Curvature = λ_min/Σλ (0 on planes, ⅓ isotropic).
- **Harris 3D corners** — normal-variation second-moment matrix + classical
  Harris response with radius non-max suppression.
- **Edges** — tangent-plane angle-gap boundary detection; principal-curvature
  linear-structure (thin rod/wire) detection.
- **Plane segmentation** — deterministic RANSAC (seeded xorshift64*) with
  least-squares refinement, greedy multi-plane extraction.
- **Descriptors** — FPFH (3×11 histogram) and SHOT-352 (4×4×2 spatial × 11
  normal-angle bins, covariant local reference frame).
- **AI-native hooks** — `LearnedDescriptor` and `ScalarCost` traits with an
  optional analytic-gradient hook for learning-agent integration.

`no_std + alloc`. Part of [tpt-perception](../../). Dual-licensed
MIT OR Apache-2.0.
