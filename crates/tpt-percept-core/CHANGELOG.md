# Changelog

All notable changes to `tpt-percept-core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the frame-safe geometric foundation
- Frame-typed `Point3D<F>`, `Vector3D<F>` with phantom frame parameters
- `Rotation3<F, T>` and `Isometry3<F, T>` for frame-safe rigid transforms
- `Twist3<F>` with exact SE(3) exponential/logarithm maps
- Weighted Kabsch and Umeyama alignment algorithms
- 3×3 symmetric eigendecomposition via cyclic Jacobi
- Cramer's rule solver for 3×3 systems
- Deterministic Xorshift64* RNG (`rng` module)
- SI unit wrappers via `tpt-math-units`
- Comprehensive `prelude` module for ergonomic imports
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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-core-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-core-0.1.0