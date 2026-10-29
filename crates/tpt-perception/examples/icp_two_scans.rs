//! Aligns two overlapping scans of the same box with point-to-point and
//! point-to-plane ICP, using a gate derived by `IcpParams::auto`.
//!
//! Run with: `cargo run -p tpt-perception --example icp_two_scans`

use tpt_perception::cloud::PointCloud;
use tpt_perception::features::normal::estimate_normals;
use tpt_perception::register::icp::{icp, IcpParams, IcpVariant};

/// Surface samples of a 2 m cube centred at the origin.
fn box_surface() -> PointCloud {
    let mut cloud = PointCloud::new();
    let steps: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.05).collect();
    for &a in &steps {
        for &b in &steps {
            for p in [
                [a, b, 1.0],
                [a, b, -1.0],
                [a, 1.0, b],
                [a, -1.0, b],
                [1.0, a, b],
                [-1.0, a, b],
            ] {
                cloud.push(p);
            }
        }
    }
    cloud
}

fn main() {
    let target = box_surface();

    // The second scan is the same box seen after a small yaw and shift.
    let (s, c) = 0.02f64.sin_cos();
    let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let translation = [0.03, -0.02, 0.01];
    let source = target.transformed_rigid(rotation, translation);

    let params = IcpParams::auto(&source, &target).expect("non-empty clouds");
    println!("auto gate: {:.3} m", params.max_correspondence_distance);

    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let point_to_point = icp(
        &source,
        &target,
        None,
        IcpVariant::PointToPoint,
        identity,
        [0.0; 3],
        &params,
    )
    .expect("point-to-point ICP");

    let normals: Vec<[f64; 3]> = estimate_normals(&target, 10)
        .expect("target has enough points")
        .iter()
        .map(|n| n.normal)
        .collect();
    let point_to_plane = icp(
        &source,
        &target,
        Some(&normals),
        IcpVariant::PointToPlane,
        identity,
        [0.0; 3],
        &params,
    )
    .expect("point-to-plane ICP");

    for (name, res) in [
        ("point-to-point", &point_to_point),
        ("point-to-plane", &point_to_plane),
    ] {
        println!(
            "{name:15} iters {:3}  converged {:5}  rmse {:.6} m  fitness {:.3}",
            res.iterations, res.converged, res.rmse, res.fitness
        );
        println!("{:15} translation {:?}", "", res.translation);
    }

    // The ground truth maps source back onto target: t* = -Rᵀ·t.
    let truth = [
        -(rotation[0][0] * translation[0] + rotation[1][0] * translation[1]),
        -(rotation[0][1] * translation[0] + rotation[1][1] * translation[1]),
        -translation[2],
    ];
    println!("ground truth    translation {truth:?}");

    if let Some(deg) = point_to_plane.degeneracy(1e-3) {
        println!("weak directions (point-to-plane): {:?}", deg.weak);
    }
}
