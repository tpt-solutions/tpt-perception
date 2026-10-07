# Changelog

All notable changes to `tpt-percept-slam` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of SLAM crate
- Monocular visual odometry: normalized 8-point essential matrix with RANSAC
- Stereo visual odometry: disparity triangulation + metric 3D-3D RANSAC/Kabsch
- LiDAR odometry: point-to-plane ICP scan matching with constant-velocity prediction
- Loop closure: Scan Context descriptors (polar grid, ring-key screening, azimuth-shift matching)
- Pose graph: dense Gauss–Newton over SE(3) with numeric Jacobians, Huber robustification
- Deterministic RANSAC via seeded Xorshift64*
- Full `no_std + alloc` support

### Changed
- N/A (initial release)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2026-10-07

Initial release.

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-slam-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-slam-0.1.0