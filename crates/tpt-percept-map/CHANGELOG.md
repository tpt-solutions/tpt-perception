# Changelog

All notable changes to `tpt-percept-map` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of spatial mapping crate
- 2D log-odds occupancy grids with Bresenham ray casting
- 3D log-odds occupancy grids with Amanatides-Woo ray casting
- Clamped, bounded log-odds updates with probability queries
- TSDF (Truncated Signed Distance Function) voxel maps
- Inverse-range weighted TSDF fusion with weighted merging
- Trilinear TSDF sampling and zero-crossing surface extraction (13-neighbour)
- Dense trilinear SDF (Signed Distance Field) grids over bounded boxes
- Numeric gradient queries for trajectory optimization
- Margin collision checking for planning
- Semantic maps with labeled point clouds
- Per-class statistics (counts, centroids, bounding boxes)
- Class and confidence filtering
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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-map-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-map-0.1.0