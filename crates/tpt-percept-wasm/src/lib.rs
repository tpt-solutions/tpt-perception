//! WebAssembly bindings for `tpt-perception`.
//!
//! Points cross the boundary as flat `Float64Array`s of `x, y, z` triples,
//! which avoids per-point JavaScript objects. Build the JavaScript package
//! with `wasm-pack build --target web` (see `README.md`).

use wasm_bindgen::prelude::*;

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::voxel::voxel_downsample;
use tpt_percept_register::icp::{icp, IcpParams, IcpVariant};

fn to_cloud(flat: &[f64]) -> Result<PointCloud, JsValue> {
    if !flat.len().is_multiple_of(3) {
        return Err(JsValue::from_str(
            "point array length must be a multiple of 3",
        ));
    }
    Ok(PointCloud::from_points(flat.as_chunks::<3>().0.to_vec()))
}

fn flatten(points: &[[f64; 3]]) -> Vec<f64> {
    points.iter().flat_map(|p| p.iter().copied()).collect()
}

/// Point-to-point ICP from the identity pose.
///
/// Returns a flat `Float64Array` laid out as:
/// `[r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz, rmse, fitness,
/// converged (0 or 1), iterations]`.
#[wasm_bindgen]
pub fn align_point_to_point(
    source: &[f64],
    target: &[f64],
    gate: f64,
) -> Result<Vec<f64>, JsValue> {
    let source = to_cloud(source)?;
    let target = to_cloud(target)?;
    let params = IcpParams {
        max_correspondence_distance: gate,
        ..IcpParams::default()
    };
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let res = icp(
        &source,
        &target,
        None,
        IcpVariant::PointToPoint,
        identity,
        [0.0; 3],
        &params,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut out: Vec<f64> = res.rotation.iter().flatten().copied().collect();
    out.extend_from_slice(&res.translation);
    out.push(res.rmse);
    out.push(res.fitness);
    out.push(if res.converged { 1.0 } else { 0.0 });
    out.push(res.iterations as f64);
    Ok(out)
}

/// Voxel-grid downsampling of a flat point array.
#[wasm_bindgen]
pub fn downsample(points: &[f64], voxel_size: f64) -> Result<Vec<f64>, JsValue> {
    let cloud = to_cloud(points)?;
    let reduced = voxel_downsample(&cloud, voxel_size)
        .ok_or_else(|| JsValue::from_str("voxel_size must be finite and > 0"))?;
    Ok(flatten(reduced.points()))
}

/// Deterministic synthetic test scene: a box surface, as a flat array. Used
/// by the demo page so it runs without uploading data.
#[wasm_bindgen]
pub fn demo_box(steps: u32) -> Vec<f64> {
    let n = steps.max(2) as usize;
    let vals: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
        .collect();
    let mut pts = Vec::with_capacity(n * n * 18);
    for &a in &vals {
        for &b in &vals {
            pts.extend_from_slice(&[
                a, b, 1.0, a, b, -1.0, a, 1.0, b, a, -1.0, b, 1.0, a, b, -1.0, a, b,
            ]);
        }
    }
    pts
}
