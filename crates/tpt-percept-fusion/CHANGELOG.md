# Changelog

All notable changes to `tpt-percept-fusion` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of multi-sensor fusion crate
- Pose timelines with bounded extrapolation and orthonormalized interpolation
- Rolling latency estimation with mean/std-dev tracking
- Self-contained dense EKF (Joseph-form update, NIS diagnostics)
- Self-contained UKF (scaled sigma-point transform, Cholesky-based sampling)
- Tsai–Lenz hand–eye extrinsic calibration from synchronized motion pairs
- Covariance intersection for consistent fusion under unknown correlation
- Chi-square innovation gating
- Consecutive-failure sensor falloff with healthy-sensor fallback
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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-fusion-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-fusion-0.1.0