# Changelog

All notable changes to `tpt-perception` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of umbrella crate
- Feature-gated re-exports of all perception sub-crates
- `default` feature set: core, cloud, features, register, slam, fusion, map
- `full` feature set: everything including verify
- Comprehensive `prelude` module with conditional re-exports
- End-to-end integration test in `tests/stack_composes`

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

[Unreleased]: https://github.com/tpt-solutions/tpt-perception/compare/tpt-perception-0.1.0...HEAD
[0.1.0]: https://github.com/tpt-solutions/tpt-perception/releases/tag/tpt-perception-0.1.0