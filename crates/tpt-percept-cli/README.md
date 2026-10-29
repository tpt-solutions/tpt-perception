# tpt-percept-cli

Command-line tool built on `tpt-perception`.

```sh
cargo install --path crates/tpt-percept-cli

tpt-percept convert    scan.pcd scan.ply            # format from extension
tpt-percept downsample scan.ply down.ply --voxel 0.2
tpt-percept register   scan_b.bin scan_a.bin          # point-to-plane ICP
tpt-percept odometry   ./kitti/sequence00 --out est.tum
```

Supported formats: PCD, PLY and KITTI `.bin` (see `tpt-percept-io`).
`downsample` drops intensity, because averaged points have no single
reflectance.
