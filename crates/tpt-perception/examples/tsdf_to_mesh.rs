//! Fuses depth points of a sphere into a TSDF from six viewpoints, extracts
//! a triangle mesh with marching tetrahedra, and writes it as OBJ.
//!
//! Run with: `cargo run -p tpt-perception --example tsdf_to_mesh`

use std::f64::consts::PI;
use std::fmt::Write as _;

use tpt_perception::map::tsdf::{TsdfMap, TsdfParams};

fn main() {
    let radius = 0.5;
    let params = TsdfParams {
        voxel_size: 0.02,
        truncation: 0.06,
        max_weight: 100.0,
    };
    let mut map = TsdfMap::new(params);

    // Viewpoints on a ring around the sphere; each sees the near half.
    for k in 0..6 {
        let angle = k as f64 * PI / 3.0;
        let origin = [2.0 * angle.cos(), 2.0 * angle.sin(), 0.0];
        let mut points = Vec::new();
        for i in 0..60 {
            for j in 0..60 {
                let theta = PI * (i as f64 + 0.5) / 60.0;
                let phi = 2.0 * PI * (j as f64 + 0.5) / 60.0;
                let p = [
                    radius * theta.sin() * phi.cos(),
                    radius * theta.sin() * phi.sin(),
                    radius * theta.cos(),
                ];
                let facing = (p[0] - origin[0]) * origin[0] + (p[1] - origin[1]) * origin[1];
                if facing < 0.0 {
                    points.push(p);
                }
            }
        }
        map.integrate_scan(origin, &points).expect("integrate");
    }

    let mesh = map.extract_mesh();
    println!(
        "{} vertices, {} triangles",
        mesh.vertices.len(),
        mesh.triangles.len()
    );
    // Every vertex should sit on the sphere, to within about a voxel.
    let worst = mesh
        .vertices
        .iter()
        .map(|v| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - radius).abs())
        .fold(0.0f64, f64::max);
    println!(
        "worst radial error: {worst:.4} m (voxel {} m)",
        params.voxel_size
    );

    let mut obj = String::new();
    for v in &mesh.vertices {
        let _ = writeln!(obj, "v {} {} {}", v[0], v[1], v[2]);
    }
    for t in &mesh.triangles {
        let _ = writeln!(obj, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1);
    }
    let out = std::env::temp_dir().join("tpt_sphere.obj");
    std::fs::write(&out, obj).expect("writable temp dir");
    println!("mesh written to {}", out.display());
}
