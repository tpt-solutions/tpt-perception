# Check whether a registration is trustworthy

A converged ICP result is not necessarily correct. Three checks catch most
bad alignments.

```rust
use tpt_perception::register::icp::IcpResult;

fn trust(result: &IcpResult) -> bool {
    // 1. Did it converge, and do enough points agree?
    if !result.converged || result.fitness < 0.5 {
        return false;
    }
    // 2. Is every direction of the pose constrained? Weak directions mean
    //    the geometry cannot tell the poses apart (corridors, flat floors).
    if let Some(deg) = result.degeneracy(1e-3) {
        if !deg.weak.is_empty() {
            return false;
        }
    }
    // 3. Is the residual small compared with the gate?
    result.rmse < 0.1
}
```

Use the covariance when a downstream filter needs it:

```rust
if let Some(cov) = result.covariance() {
    // cov is rmse² · (AᵀWA)⁻¹ in (translation, rotation) order.
    // Treat it as relative uncertainty; see the IcpResult::covariance docs.
    let _ = cov;
}
```

The thresholds above are starting points. Calibrate them on data from your
own sensor.
