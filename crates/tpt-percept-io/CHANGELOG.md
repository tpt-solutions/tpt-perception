# Changelog

All notable changes to `tpt-percept-io` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the file I/O crate
- PCD readers and writers (ascii, binary; `binary_compressed` rejected)
- PLY readers and writers (ascii, binary little-endian)
- KITTI odometry `.bin` readers and writers
- TUM trajectory readers and writers
- Bit-exact `TPTREC01` recordings for deterministic replay
- Nominal sensor presets (Velodyne HDL-64E, VLP-16, Ouster OS1-64)

### Changed
- N/A (initial release)

### Deprecated
- N/A
