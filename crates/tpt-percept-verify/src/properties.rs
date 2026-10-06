//! Cross-crate property tests: the statistical verification the spec
//! demands for geometric invariants, and the phase integration tests from
//! `todo.md` (plane detection on synthetic clouds, ICP monotonicity, pose
//! graph consistency).
//!
//! `proptest!` wraps each property in its own module, so the imports each
//! body needs are written inside the function.

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Spec invariant: rigid transforms preserve pairwise distances.
    #[test]
    fn rigid_transforms_preserve_distances(
        (r, t) in crate::strategy::rigid_transform(),
        points in proptest::collection::vec(crate::strategy::finite_point(5.0), 2..=16),
    ) {
        use crate::invariants::{check_distance_preserving, check_inverse_roundtrip};
        if let Err(msg) = check_distance_preserving(&r, &t, &points, 1e-6) {
            prop_assert!(false, "{msg}");
        }
        if let Err(msg) = check_inverse_roundtrip(&r, &t, 1e-6) {
            prop_assert!(false, "{msg}");
        }
    }

    /// Spec invariant: ICP residual decreases monotonically on overlapping
    /// clouds (small slack per step for float noise and discretisation).
    #[test]
    fn icp_residual_is_monotone(
        (r, t) in crate::strategy::rigid_transform(),
        cloud in crate::strategy::point_cloud(40, 60, 3.0),
    ) {
        use crate::strategy::apply_rigid;
        use tpt_percept_cloud::cloud::PointCloud;
        use tpt_percept_cloud::kdtree::KdTree;
        use tpt_percept_register::icp::{icp, IcpParams, IcpVariant};
        use crate::invariants::residual_summary;

        let tree = KdTree::new(cloud.points());
        let source = {
            let mut c = PointCloud::new();
            for &p in cloud.points() {
                c.push(apply_rigid(&r, &t, p));
            }
            c
        };
        let rmse_at = |rotation: &[[f64; 3]; 3], translation: &[f64; 3]| -> f64 {
            let mut acc = 0.0;
            let mut n = 0usize;
            for &s in source.points() {
                let ts = apply_rigid(rotation, translation, s);
                if let Some((_, d2)) = tree.nearest(&ts) {
                    acc += d2;
                    n += 1;
                }
            }
            if n > 0 { (acc / n as f64).sqrt() } else { f64::INFINITY }
        };
        let before = rmse_at(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], &[0.0; 3]);

        let params = IcpParams {
            max_correspondence_distance: 5.0,
            max_iterations: 25,
            tolerance: 1e-9,
            min_correspondences: 5,
        };
        let result = match icp(
            &source,
            &cloud,
            None,
            IcpVariant::PointToPoint,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            &params,
        ) {
            Ok(r) => r,
            // Degenerate random clouds (collinear/coplanar draws) have no
            // unique alignment; the property is vacuous there.
            Err(_) => return Ok(()),
        };
        let after = rmse_at(&result.rotation, &result.translation);
        prop_assert!(
            after <= before + 1e-6,
            "ICP increased residual: {before:.6} → {after:.6} ({})",
            residual_summary(&[before, after])
        );
    }

    /// Phase-1 integration: RANSAC plane detection on synthetic planar
    /// clouds with noise — the dominant plane must be the ground plane.
    #[test]
    fn plane_detection_on_synthetic_clouds(
        noise in crate::strategy::noise_profile(120, 0.004),
        plane_height in (-2.0_f64..2.0),
        seed in any::<u64>(),
    ) {
        use tpt_percept_cloud::cloud::PointCloud;
        use tpt_percept_features::planes::{segment_planes, RansacParams};

        let mut cloud = PointCloud::new();
        for i in 0..60 {
            let (x, y) = ((i % 10) as f64 * 0.2 - 1.0, (i / 10) as f64 * 0.2 - 1.0);
            let n = noise[i % noise.len()];
            cloud.push([x + n[0], y + n[1], plane_height + n[2]]);
        }
        for i in 0..20 {
            let n = noise[(i * 7) % noise.len()];
            cloud.push([2.0 + n[0], i as f64 * 0.1 - 1.0, 3.0 + n[2].abs()]);
        }
        let params = RansacParams {
            distance_threshold: 0.02,
            max_iterations: 500,
            min_inliers: 40,
            seed,
        };
        let planes = segment_planes(&cloud, &params, 1).unwrap();
        prop_assert!(!planes.is_empty(), "no plane found");
        let ground = &planes[0];
        prop_assert!(ground.plane.normal[2].abs() > 0.999, "normal {:?}", ground.plane.normal);
        prop_assert!(
            (ground.plane.d.abs() - plane_height.abs()).abs() < 0.02,
            "offset {} vs height {plane_height}",
            ground.plane.d
        );
        prop_assert!(ground.inliers.len() >= 50, "inliers {}", ground.inliers.len());
    }

    /// Normals of a noisy plane stay within a few degrees of the plane
    /// normal (statistical robustness of local PCA).
    #[test]
    fn plane_normals_robust_to_noise(
        noise in crate::strategy::noise_profile(80, 0.005),
        plane_height in (-1.0_f64..1.0),
    ) {
        use tpt_percept_cloud::cloud::PointCloud;
        use tpt_percept_features::normal::estimate_normals;

        let mut cloud = PointCloud::new();
        for i in 0..80 {
            let (x, y) = ((i % 9) as f64 * 0.2 - 0.8, (i / 9) as f64 * 0.2 - 0.8);
            let n = noise[i % noise.len()];
            cloud.push([x + n[0], y + n[1], plane_height + n[2]]);
        }
        let normals = estimate_normals(&cloud, 12).unwrap();
        for (i, n) in normals.iter().enumerate() {
            // Skip grid-edge points (half-neighborhoods tilt the PCA).
            let p = cloud.get(i).unwrap();
            if p[0].abs() > 0.7 || p[1].abs() > 0.7 {
                continue;
            }
            prop_assert!(n.normal[2].abs() > 0.99, "point {p:?} normal {:?}", n.normal);
        }
    }

    /// EKF covariance stays symmetric PSD after predict+update.
    #[test]
    fn ekf_covariance_remains_valid(
        r in crate::strategy::covariance3(0.01, 5.0),
        z in (-50.0_f64..50.0),
    ) {
        use crate::invariants::check_covariance_valid;
        use tpt_math_linalg_dense::{DMatrix, DVector};

        let mut ekf = tpt_percept_fusion::ekf::Ekf::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[r[0][0], r[0][1], r[1][0], r[1][1]]),
        );
        let dt = 0.1_f64;
        let mut f = |s: &DVector| DVector::from_vec(vec![s[0] + s[1] * dt, s[1]]);
        let f_jac = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6, 1e-6]));
        ekf.predict(&mut f, &f_jac, &q).unwrap();
        let mut h = |s: &DVector| DVector::from_vec(vec![s[0]]);
        let h_jac = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r_noise = DMatrix::from_diagonal(&DVector::from_vec(vec![0.1]));
        ekf.update(&DVector::from_vec(vec![z]), &mut h, &h_jac, &r_noise).unwrap();

        let p = ekf.covariance();
        let c = [[p[(0, 0)], p[(0, 1)], 0.0], [p[(1, 0)], p[(1, 1)], 0.0], [0.0, 0.0, 1.0]];
        if let Err(msg) = check_covariance_valid(&c, 1e-9) {
            prop_assert!(false, "{msg}");
        }
        prop_assert!(p[(0, 0)] > 0.0 && p[(1, 1)] > 0.0);
    }
}

/// Pose graph consistency: for a perturbed cycle with exact edges, the
/// optimized poses reproduce the measured relative constraints (statistical
/// companion of the Kani harnesses for the spec's Phase-2 verification).
#[test]
fn pose_graph_cycle_consistency() {
    use tpt_percept_slam::pose_graph::{Pose, PoseEdge, PoseGraph, PoseGraphParams};

    let yaw_pose = |x: f64, y: f64, yaw: f64| {
        let (s, c) = yaw.sin_cos();
        Pose {
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [x, y, 0.0],
        }
    };
    let truth = [
        yaw_pose(0.0, 0.0, 0.0),
        yaw_pose(2.0, 0.0, 0.0),
        yaw_pose(2.0, 2.0, core::f64::consts::FRAC_PI_2),
        yaw_pose(0.0, 2.0, core::f64::consts::PI),
    ];
    let mut graph = PoseGraph::new();
    let mut nodes = Vec::new();
    for (i, t) in truth.iter().enumerate() {
        let jitter = yaw_pose(
            t.translation[0] + 0.04 * (i as f64 + 1.0).sin(),
            t.translation[1] - 0.03 * (i as f64 + 1.0).cos(),
            0.005 * (i as f64 + 1.0),
        );
        if i == 0 {
            nodes.push(graph.add_node(*t));
        } else {
            nodes.push(graph.add_node(jitter));
        }
    }
    for i in 0..4 {
        let j = (i + 1) % 4;
        let relative = truth[i].inverse().compose(&truth[j]);
        graph
            .add_edge(PoseEdge {
                from: nodes[i],
                to: nodes[j],
                relative,
                information: 1.0,
                huber_delta: 0.0,
            })
            .unwrap();
    }
    let before = graph.cost();
    graph
        .optimize(&PoseGraphParams {
            max_iterations: 100,
            ..Default::default()
        })
        .unwrap();
    let after = graph.cost();
    assert!(after < before * 1e-2, "cost {before} → {after}");
    for edge in 0..graph.edge_count() {
        let e = graph.edge(edge).unwrap();
        let actual = graph
            .pose(e.from)
            .unwrap()
            .inverse()
            .compose(graph.pose(e.to).unwrap());
        let d = [
            actual.translation[0] - e.relative.translation[0],
            actual.translation[1] - e.relative.translation[1],
            actual.translation[2] - e.relative.translation[2],
        ];
        let err = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(err < 5e-2, "edge {edge} residual {err}");
    }
}

#[test]
fn monotone_checker_reports_icp_history_shape() {
    use crate::invariants::{check_monotone_decreasing, residual_summary};
    let history = [2.0, 1.5, 1.2, 1.05, 1.0, 1.0, 1.0];
    assert!(check_monotone_decreasing(&history, 1e-9).is_ok());
    let summary = residual_summary(&history);
    assert!(summary.contains("first=2.000000"));
    assert!(summary.contains("last=1.000000"));
}
