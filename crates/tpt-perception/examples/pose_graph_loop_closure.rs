//! Corrects odometry drift around a square with one loop-closure edge.
//!
//! Run with: `cargo run -p tpt-perception --example pose_graph_loop_closure`

use tpt_perception::slam::pose_graph::{Pose, PoseEdge, PoseGraph, PoseGraphParams};

fn translate(x: f64, y: f64) -> Pose {
    Pose {
        translation: [x, y, 0.0],
        ..Pose::identity()
    }
}

fn main() {
    // Ground truth: a 4 m square driven once, returning to the start.
    let corners = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];

    // Odometry drifts: each leg carries a small constant bias. A pure scale
    // error would cancel around a closed loop, so a bias is what makes the
    // chain's last node miss the start.
    let mut graph = PoseGraph::new();
    let mut estimate = Pose::identity();
    graph.add_node(estimate);
    for leg in corners.windows(2) {
        let (dx, dy) = (leg[1][0] - leg[0][0], leg[1][1] - leg[0][1]);
        let measured = translate(dx + 0.1, dy + 0.05);
        let from = graph.len() - 1;
        estimate = estimate.compose(&measured);
        graph.add_node(estimate);
        graph
            .add_edge(PoseEdge {
                from,
                to: from + 1,
                relative: measured,
                information: 1.0,
                information_matrix: None,
                huber_delta: 0.0,
            })
            .expect("valid edge");
    }

    // Loop closure: the place recognised at the end is the start.
    graph
        .add_edge(PoseEdge {
            from: 4,
            to: 0,
            relative: translate(0.0, 0.0),
            information: 50.0,
            information_matrix: None,
            huber_delta: 0.0,
        })
        .expect("valid edge");

    let end_before = graph.pose(4).expect("node 4").translation;
    let report = graph
        .optimize(&PoseGraphParams::default())
        .expect("optimisation runs");
    let end_after = graph.pose(4).expect("node 4").translation;

    println!(
        "drifted end pose:   [{:.3}, {:.3}]",
        end_before[0], end_before[1]
    );
    println!(
        "corrected end pose: [{:.3}, {:.3}]",
        end_after[0], end_after[1]
    );
    println!(
        "cost {:.3e} -> {:.3e} in {} iterations (converged: {})",
        report.initial_cost, report.final_cost, report.iterations, report.converged
    );
}
