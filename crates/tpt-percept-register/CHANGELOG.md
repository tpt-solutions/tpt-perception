# Changelog

All notable changes to `tpt-percept-register` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of point cloud registration crate
- ICP (Iterative Closest Point): point-to-point (Kabsch) and point-to-plane (Chen & Medioni) variants
- Shared convergence contract with fitness/RMSE reporting
- NDT (Normal Distributions Transform): voxel-Gaussian map + Gauss–Newton registration
- Feature-based registration: descriptor ratio matching + RANSAC/Kabsch
- Global registration: branch-and-bound over (x, y, yaw) with valid lower bound on trimmed residuals
- Loop closure detection: yaw Hough voting over normal-azimuth pairs + translation RANSAC
- Deterministic algorithms via seeded Xorshift64*
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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-register-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-register-0.1.0