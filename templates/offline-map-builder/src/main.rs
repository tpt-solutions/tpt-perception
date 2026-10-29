//! Builds a triangle mesh of the scene from a sequence of KITTI scans.
//!
//! Each scan is placed in the world with LiDAR odometry, then fused into a
//! TSDF whose samples come from the sensor position. The mesh is written as
//! OBJ.
//!
//! Usage: `{{project-name}} <scan-dir> <out.obj>`

use std::fmt::Write as _;
use std::path::PathBuf;

use tpt_percept_io::read_kitti_bin;
use tpt_perception::map::tsdf::{TsdfMap, TsdfParams};
use tpt_perception::slam::lidar_odometry::{LidarOdometry, LidarOdometryParams};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: {{project-name}} <scan-dir> <out.obj>");
        std::process::exit(2);
    };

    let mut scans: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("readable scan directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    scans.sort();

    // Voxel size and truncation trade detail against noise. Tune per sensor.
    let mut map = TsdfMap::new(TsdfParams {
        voxel_size: 0.1,
        truncation: 0.3,
        max_weight: 100.0,
    });
    let mut odometry = LidarOdometry::new(LidarOdometryParams::default());

    for path in &scans {
        let scan = read_kitti_bin(path).expect("valid scan").cloud;
        let step = odometry.process(&scan).expect("odometry step");
        // Scan points are in the sensor frame; move them to the world with
        // the odometry pose, and use the sensor position as the origin.
        let world = scan.transformed_rigid(step.rotation, step.translation);
        map.integrate_scan(step.translation, world.points())
            .expect("integrate scan");
    }

    let mesh = map.extract_mesh();
    let mut text = String::new();
    for v in &mesh.vertices {
        let _ = writeln!(text, "v {} {} {}", v[0], v[1], v[2]);
    }
    for t in &mesh.triangles {
        let _ = writeln!(text, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1);
    }
    std::fs::write(&out, text).expect("writable output path");
    println!(
        "{} scans → {} vertices, {} triangles in {out}",
        scans.len(),
        mesh.vertices.len(),
        mesh.triangles.len()
    );
}
