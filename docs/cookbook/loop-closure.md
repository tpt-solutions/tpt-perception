# Correct drift with loop closure

Goal: when the sensor returns to a place it has seen, add a constraint that
pulls the trajectory back into shape.

1. Detect the revisit, for example with `slam::loop_closure::ScanContext`
   (`detect_loop`).
2. Register the two scans to get the relative pose between the earlier and
   current node, using `register::icp`.
3. Add an edge with that relative pose. Weight it with the registration
   covariance if you have it:

```rust
use tpt_perception::slam::pose_graph::{Pose, PoseEdge, PoseGraph, PoseGraphParams};

fn close_loop(graph: &mut PoseGraph, from: usize, to: usize, relative: Pose, information: f64) {
    graph
        .add_edge(PoseEdge {
            from,
            to,
            relative,
            information,
            information_matrix: None, // or Some(6x6) from the registration
            huber_delta: 1.0,         // guards against a false loop
        })
        .expect("valid edge");
    let report = graph.optimize(&PoseGraphParams::default()).expect("optimise");
    if !report.converged {
        eprintln!("pose graph did not converge; cost {:.3e}", report.final_cost);
    }
}
```

Always check `OptimizeReport::converged`. A non-converged graph has not
been corrected.

Runnable version: `cargo run -p tpt-perception --example pose_graph_loop_closure`.
