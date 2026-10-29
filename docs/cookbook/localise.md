# Localise against a fixed map

Goal: you have a map (a point cloud) and each new scan should be placed in
it. Build the target once, then reuse it for every scan.

```rust
use tpt_perception::cloud::PointCloud;
use tpt_perception::register::icp::{icp_prepared, IcpParams, IcpVariant, PreparedTarget};

fn localise(map: PointCloud, scans: &[PointCloud]) -> Result<(), Box<dyn std::error::Error>> {
    // Build the k-d tree and normals once.
    let normals = tpt_perception::features::normal::estimate_normals(&map, 10)
        .ok_or("map too small")?
        .iter()
        .map(|n| n.normal)
        .collect();
    let target = PreparedTarget::new(map, Some(normals))?;

    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // Set the gate from the first scan against the map.
    let params = IcpParams::auto(&scans[0], target.cloud()).ok_or("empty scan")?;

    for scan in scans {
        let result = icp_prepared(scan, &target, IcpVariant::PointToPlane, identity, [0.0; 3], &params)?;
        println!("fitness {:.2}, translation {:?}", result.fitness, result.translation);
    }
    Ok(())
}
```

Seed each call with the previous pose (or your odometry prediction) instead
of the identity. Identity is only safe when the scans already overlap.

Runnable version: `cargo run -p tpt-perception --example icp_two_scans`.
