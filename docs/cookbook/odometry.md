# Estimate LiDAR odometry

Goal: turn a stream of scans into a trajectory.

```rust
use tpt_perception::slam::lidar_odometry::{LidarOdometry, LidarOdometryParams};

fn run(scans: impl Iterator<Item = tpt_perception::cloud::PointCloud>) {
    let mut odometry = LidarOdometry::new(LidarOdometryParams {
        map_window: 5,       // scans kept in the local map
        predict_motion: true, // constant-velocity seed for ICP
        ..LidarOdometryParams::default()
    });
    for scan in scans {
        match odometry.process(&scan) {
            Ok(step) if step.converged => {
                println!("t = {:?}", step.translation);
            }
            Ok(_) => eprintln!("frame did not converge; pose is a best guess"),
            Err(e) => eprintln!("skipped frame: {e}"),
        }
    }
}
```

Tuning:

- Raise `map_window` for slow motion or a sparse sensor. Lower it when
  the environment changes quickly.
- `icp.max_correspondence_distance` (default 0.5 m) should be larger than
  the per-frame motion. Use `IcpParams::auto` to derive it from data.
- Scan-to-map odometry still drifts. Pair it with loop closure to correct
  drift over long runs.

Runnable version: `cargo run -p tpt-perception --example kitti_lidar_odometry`.
