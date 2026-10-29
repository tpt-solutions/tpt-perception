//! Segments the dominant planes of a synthetic room (floor and two walls)
//! with seeded RANSAC.
//!
//! Run with: `cargo run -p tpt-perception --example plane_segmentation`

use tpt_perception::cloud::PointCloud;
use tpt_perception::features::planes::{segment_planes, RansacParams};

fn main() {
    let mut cloud = PointCloud::new();
    // Floor at z = 0.
    for x in 0..60 {
        for y in 0..60 {
            cloud.push([x as f64 * 0.05, y as f64 * 0.05, 0.0]);
        }
    }
    // Wall at x = 3.
    for y in 0..60 {
        for z in 0..40 {
            cloud.push([3.0, y as f64 * 0.05, z as f64 * 0.05]);
        }
    }
    // Wall at y = 3.
    for x in 0..60 {
        for z in 0..40 {
            cloud.push([x as f64 * 0.05, 3.0, z as f64 * 0.05]);
        }
    }
    // Clutter that belongs to no plane.
    for i in 0..50 {
        let f = i as f64;
        cloud.push([1.0 + 0.01 * f.sin(), 1.0 + 0.02 * f.cos(), 0.5 + 0.01 * f]);
    }

    let params = RansacParams {
        distance_threshold: 0.02,
        ..RansacParams::default()
    };
    let segments = segment_planes(&cloud, &params, 3).expect("valid parameters");

    println!("{} points, {} planes found", cloud.len(), segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let n = seg.plane.normal;
        println!(
            "plane {i}: normal ({:+.2}, {:+.2}, {:+.2})  d {:+.3}  inliers {}",
            n[0],
            n[1],
            n[2],
            seg.plane.d,
            seg.inliers.len()
        );
    }
}
