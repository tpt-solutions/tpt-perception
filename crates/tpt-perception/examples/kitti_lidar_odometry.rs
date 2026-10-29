//! LiDAR odometry over a sequence of scans.
//!
//! With a directory argument it reads KITTI `.bin` scans in name order.
//! Without one it simulates a short straight drive so the example runs
//! offline.
//!
//! Run with: `cargo run -p tpt-perception --example kitti_lidar_odometry -- <dir>`

use std::path::PathBuf;

use tpt_percept_io::{read_kitti_bin, write_tum_to, TumPose};
use tpt_perception::cloud::PointCloud;
use tpt_perception::slam::lidar_odometry::{LidarOdometry, LidarOdometryParams};

/// A rolling-terrain scene: a gently bumpy ground plus two pillars.
fn scene() -> PointCloud {
    let mut cloud = PointCloud::new();
    for x in -40..=40 {
        for y in -40..=40 {
            let (xf, yf) = (x as f64 * 0.25, y as f64 * 0.25);
            cloud.push([xf, yf, 0.1 * (0.3 * xf).sin() + 0.1 * (0.2 * yf).cos()]);
        }
    }
    for h in 0..20 {
        for &(px, py) in &[(3.0, 2.0), (-4.0, -3.0)] {
            cloud.push([px, py, 0.1 * h as f64]);
        }
    }
    cloud
}

fn main() {
    let scans: Vec<PointCloud> = match std::env::args().nth(1) {
        Some(dir) => {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
                .expect("readable scan directory")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "bin"))
                .collect();
            paths.sort();
            paths
                .iter()
                .map(|p| read_kitti_bin(p).expect("valid KITTI scan").cloud)
                .collect()
        }
        None => {
            // The sensor advances 0.5 m per frame along x.
            let base = scene();
            (0..10)
                .map(|k| {
                    let shift = -0.5 * k as f64;
                    base.transformed_rigid(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [shift, 0.0, 0.0],
                    )
                })
                .collect()
        }
    };

    let mut odom = LidarOdometry::new(LidarOdometryParams::default());
    let mut trajectory = Vec::new();
    for (k, scan) in scans.iter().enumerate() {
        let step = odom.process(scan).expect("odometry step");
        println!(
            "frame {k:3}  t = [{:+.3}, {:+.3}, {:+.3}]  converged {}",
            step.translation[0], step.translation[1], step.translation[2], step.converged
        );
        trajectory.push(TumPose {
            timestamp: k as f64 * 0.1,
            translation: step.translation,
            quaternion: [0.0, 0.0, 0.0, 1.0],
        });
    }

    let out = std::env::temp_dir().join("tpt_odometry_example.tum");
    let file = std::fs::File::create(&out).expect("writable temp dir");
    write_tum_to(file, &trajectory).expect("trajectory written");
    println!("trajectory written to {}", out.display());
}
