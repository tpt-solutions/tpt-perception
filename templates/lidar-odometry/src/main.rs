//! LiDAR odometry over a directory of KITTI `.bin` scans.
//!
//! Usage: `{{project-name}} <scan-dir> <trajectory.tum>`

use std::path::PathBuf;

use tpt_percept_io::{read_kitti_bin, write_tum_to, TumPose};
use tpt_perception::slam::lidar_odometry::{LidarOdometry, LidarOdometryParams};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: {{project-name}} <scan-dir> <trajectory.tum>");
        std::process::exit(2);
    };

    let mut scans: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("readable scan directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    scans.sort();

    // Tune here: the defaults suit roughly 10 Hz spinning LiDARs.
    let mut odometry = LidarOdometry::new(LidarOdometryParams::default());
    let mut trajectory = Vec::with_capacity(scans.len());
    for (k, path) in scans.iter().enumerate() {
        let scan = read_kitti_bin(path).expect("valid scan").cloud;
        let step = odometry.process(&scan).expect("odometry step");
        trajectory.push(TumPose {
            timestamp: k as f64 * 0.1,
            translation: step.translation,
            // Orientation is omitted here for brevity; see the `tpt-percept`
            // CLI for a full rotation-to-quaternion conversion.
            quaternion: [0.0, 0.0, 0.0, 1.0],
        });
    }

    let file = std::fs::File::create(&out).expect("writable output path");
    write_tum_to(file, &trajectory).expect("trajectory written");
    println!("{} poses written to {out}", trajectory.len());
}
