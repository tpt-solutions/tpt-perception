# tpt-percept-py

Python bindings for `tpt-perception`, built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs). This is the only crate in the workspace
that uses a foreign-function interface. The algorithm crates stay FFI-free.

## Build and install

```sh
cd crates/tpt-percept-py
python -m pip install maturin
maturin develop --release        # install into the active environment
# or build a wheel:
maturin build --release
```

## Use

```python
import tpt_perception as tp

target = [[x * 0.05, y * 0.05, 1.0] for x in range(20) for y in range(20)]
source = [[p[0] + 0.01, p[1], p[2]] for p in target]

result = tp.align_icp(source, target, max_correspondence_distance=0.25)
print(result["translation"], result["rmse"], result["converged"])

odom = tp.LidarOdometry(map_window=5)
rotation, translation, converged = odom.process(target)

planes = tp.segment_planes(target, distance_threshold=0.02, max_planes=2)
points = tp.read_kitti_points("scans/000000.bin")
```

Points are plain lists of `[x, y, z]`. No NumPy dependency is required.

## Tests

```sh
python -m unittest discover -s tests -v
```

The tests need the extension installed (`maturin develop`).
