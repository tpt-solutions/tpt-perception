# Changelog

All notable changes to `tpt-percept-verify` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of verification harnesses crate
- Proptest strategies: valid points, exact rotations, rigid transforms, point clouds, noise profiles, PSD covariances
- Invariant checkers: distance preservation, inverse roundtrips, monotone residuals, covariance validity
- Kani harnesses: voxel-key safety, log-odds clamping bounds, Kabsch degenerate sets, frame transform roundtrips
- Property tests: rigid transform distance preservation, ICP monotonicity, plane detection, normal robustness, EKF covariance, pose graph consistency
- Full `std` support (required for proptest/Kani)

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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-verify-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-verify-0.1.0