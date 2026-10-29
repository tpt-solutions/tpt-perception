# Build a surface map from depth

Goal: fuse depth returns from several viewpoints into one surface and export
a triangle mesh.

```rust
use tpt_perception::map::tsdf::{TsdfMap, TsdfParams};

fn fuse(views: &[([f64; 3], Vec<[f64; 3]>)]) -> Result<(), Box<dyn std::error::Error>> {
    let mut map = TsdfMap::new(TsdfParams {
        voxel_size: 0.02,
        truncation: 0.06,  // about three voxels
        max_weight: 100.0,
    });
    for (origin, points) in views {
        // Points are in the world frame; `origin` is the sensor position.
        map.integrate_scan(*origin, points)?;
    }
    let mesh = map.extract_mesh();
    println!("{} vertices, {} triangles", mesh.vertices.len(), mesh.triangles.len());
    Ok(())
}
```

Guidance:

- `truncation` sets how far from the surface a measurement can influence a
  voxel. Smaller values give sharper detail but need more views to fill in.
- Expect vertex error of up to a few voxels near thin features and at the
  edge of the observed region. The `tsdf_to_mesh` example reports this.
- The mesh winding is not guaranteed consistent. Recompute normals from
  triangles if you need them.

Runnable version: `cargo run -p tpt-perception --example tsdf_to_mesh`.
