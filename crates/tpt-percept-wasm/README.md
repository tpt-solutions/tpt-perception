# tpt-percept-wasm

WebAssembly bindings and a browser demo for `tpt-perception`.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build crates/tpt-percept-wasm --target web
# then serve the crate directory so web/index.html can load ./pkg:
cd crates/tpt-percept-wasm && python -m http.server 8000
# open http://localhost:8000/web/index.html
```

`cargo check -p tpt-percept-wasm --target wasm32-unknown-unknown` checks the
crate without `wasm-pack`.

The exported functions take flat `Float64Array`s of `x, y, z` triples:

- `align_point_to_point(source, target, gate)` returns the rotation, translation,
  RMSE, fitness, convergence flag and iteration count.
- `downsample(points, voxel_size)` returns the voxel centroids.
- `demo_box(steps)` returns a synthetic test surface.
