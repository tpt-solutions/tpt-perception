//! Builds a 2-D occupancy grid from three sensor sweeps against a wall and
//! prints it as ASCII.
//!
//! Run with: `cargo run -p tpt-perception --example occupancy_from_scans`

use tpt_perception::map::occupancy::{OccupancyGrid2D, OccupancyParams};

fn main() {
    let params = OccupancyParams {
        voxel_size: 0.25,
        ..OccupancyParams::default()
    };
    let mut grid = OccupancyGrid2D::new([-4.0, -4.0], params);

    // The sensor sits at three poses, each sweeping a 180° arc. A wall runs
    // along y = 2 m; rays stop there or at a 6 m maximum range.
    for origin in [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]] {
        let mut hits = Vec::new();
        for i in 0..=60 {
            let angle = std::f64::consts::PI * (i as f64 / 60.0);
            let (s, c) = angle.sin_cos();
            let range = if s > 1e-6 { (2.0 / s).min(6.0) } else { 6.0 };
            hits.push([origin[0] + range * c, origin[1] + range * s]);
        }
        grid.insert_scan(origin, &hits);
    }

    for row in (-16..=16).rev() {
        let mut line = String::new();
        for col in -16..=16 {
            let p = grid.probability_at([col as f64 * 0.25, row as f64 * 0.25]);
            line.push(match p {
                p if p > 0.6 => '#',
                p if p < 0.4 => '.',
                _ => '?',
            });
        }
        println!("{line}");
    }
}
