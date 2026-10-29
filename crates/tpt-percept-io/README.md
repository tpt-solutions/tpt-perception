# tpt-percept-io

Reading and writing point clouds and trajectories for `tpt-perception`.

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PCD (PCL) | ascii, binary | ascii, binary | `x y z` + optional `intensity`; `binary_compressed` rejected |
| PLY | ascii, binary little-endian | ascii, binary little-endian | vertex element only; list properties rejected |
| KITTI odometry `.bin` | yes | yes | little-endian `f32` `x y z reflectance` |
| TUM trajectory | yes | yes | `timestamp tx ty tz qx qy qz qw` |

```rust
use tpt_percept_io::{read_pcd, read_kitti_bin};

let scan = read_kitti_bin("scans/000000.bin")?;
println!("{} points", scan.cloud.len());
```

Intensity is returned alongside the geometry in `LoadedCloud::intensity`,
because `PointCloud` stores positions only.
