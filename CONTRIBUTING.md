# Contributing to tpt-perception

Thanks for helping build the TPT perception stack.

## Ground rules (from spec.txt — these are enforced)

1. **No external algorithm dependencies.** No PCL/Open3D/OpenCV FFI, no
   `nalgebra`/`faer` (Apache-2.0-only wrap targets, ADR-0007). Math comes
   from in-house `tpt-math` crates (path deps) or is implemented here.
2. **Licensing.** Everything is MIT OR Apache-2.0. New dependencies must be
   in the `deny.toml` allow-list (`cargo deny check` must stay clean).
3. **Frame safety.** Public APIs that touch geometry must be frame-typed
   (`Point3D<F>`, `Isometry3<From, To>`). A raw `[f64; 3]` is acceptable
   only inside an algorithm's internals.
4. **Mathematical contracts.** Document geometric invariants in rustdoc
   (units, frames, convergence conditions) and cover them with tests:
   unit tests for exact cases, `proptest` for statistical invariants,
   Kani harnesses (`#[cfg(kani)]`) for bounded proofs of critical paths.
5. **`no_std` + `alloc`.** Algorithm crates must build with
   `--no-default-features --features alloc`. No `std`-only APIs outside
   `std`-gated modules.

## Checklist for every change

```sh
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --no-deps
cargo deny check
```

- Rustdoc on all public items (`missing_docs` is warn-by-default and
  treated as an error in CI).
- Unit tests alongside the code they cover.

## Conventions

- Units are metres / radians / seconds as `f64` in algorithm hot paths;
  unit-safe types (`tpt-math-units`) at API boundaries where mixing is a
  realistic risk (timestamps, voxel sizes, sensor rates).
- Active rotations, column vectors, right-handed frames (matches
  `tpt-math-geometry` conventions, see its crate docs).
- Errors: `#[non_exhaustive]` enums implementing `core::error::Error`;
  no panics on invalid input from public APIs (return `Result`).

## Commit style

Short imperative subject, blank line, body explaining *why* when not
obvious. Reference todo.md items in the body when a change completes a
tracked phase item.

## License

By contributing you agree your contributions are dual-licensed
MIT OR Apache-2.0.
