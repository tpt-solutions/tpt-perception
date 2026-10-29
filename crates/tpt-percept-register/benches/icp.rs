#![allow(missing_docs)]

//! ICP throughput on a synthetic box, point-to-point and point-to-plane.
//!
//! Run with: `cargo bench -p tpt-percept-register`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_features::normal::estimate_normals;
use tpt_percept_register::icp::{icp, IcpParams, IcpVariant};

fn box_surface() -> PointCloud {
    let mut cloud = PointCloud::new();
    let steps: Vec<f64> = (-40..=40).map(|i| i as f64 * 0.025).collect();
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

fn bench_icp(c: &mut Criterion) {
    let target = box_surface();
    let source = target.transformed_rigid(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.02, -0.01, 0.005],
    );
    let normals: Vec<[f64; 3]> = estimate_normals(&target, 10)
        .expect("enough points")
        .iter()
        .map(|n| n.normal)
        .collect();
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let params = IcpParams::auto(&source, &target).expect("non-empty");

    c.bench_function("icp/point_to_point", |b| {
        b.iter(|| {
            icp(
                black_box(&source),
                black_box(&target),
                None,
                IcpVariant::PointToPoint,
                identity,
                [0.0; 3],
                &params,
            )
            .expect("icp runs")
        })
    });
    c.bench_function("icp/point_to_plane", |b| {
        b.iter(|| {
            icp(
                black_box(&source),
                black_box(&target),
                Some(&normals),
                IcpVariant::PointToPlane,
                identity,
                [0.0; 3],
                &params,
            )
            .expect("icp runs")
        })
    });
}

criterion_group!(benches, bench_icp);
criterion_main!(benches);
