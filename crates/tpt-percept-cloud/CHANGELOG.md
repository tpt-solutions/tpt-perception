# Changelog

All notable changes to `tpt-percept-cloud` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of point cloud processing crate
- `PointCloud` and frame-tagged `FrameCloud<F>` containers
- Sparse voxel grid hashing with panic-free key computation
- Voxel-grid downsampling by centroid
- Statistical outlier removal (kNN mean-distance threshold)
- Radius outlier removal filter
- Pass-through cropping filter (all axes)
- Balanced k-d tree for exact k-NN and radius queries
- Region octree for hierarchical ball/box queries
- Static R-tree (STR-style bulk load) for AABB indexing
- Streaming pipeline with sliding window aggregator
- Composable filter stages (voxel, crop)
- All filters return survivor indices for attribute preservation
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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-percept-cloud-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-percept-cloud-0.1.0