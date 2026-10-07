# Changelog

All notable changes to `tpt-percept-features` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of geometric feature extraction crate
- PCA-based normal estimation with curvature computation
- Viewpoint-oriented normal alignment
- 3D Harris corner detection with non-max suppression
- Boundary edge detection (tangent-plane angle-gap)
- Crease edge detection (principal curvature ridges)
- Deterministic RANSAC plane segmentation with LS refinement
- FPFH descriptors (33-bin histogram)
- SHOT-352 descriptors with covariant local reference frame
- `LearnedDescriptor` and `ScalarCost` traits for AI-native integration
- `DescriptorContext` for passing auxiliary data to learned descriptors
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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-features-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-features-0.1.0