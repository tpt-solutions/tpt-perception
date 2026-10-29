//! `tpt-percept` — command-line access to the perception stack.
//!
//! ```text
//! tpt-percept convert    <in> <out> [--ascii]
//! tpt-percept downsample <in> <out> --voxel <metres>
//! tpt-percept register   <source> <target> [--point-to-point]
//! tpt-percept odometry   <scan-dir> [--out <trajectory.tum>]
//! ```
//!
//! Formats are chosen from the file extension: `.pcd`, `.ply`, `.bin`
//! (KITTI). `register` and `odometry` print results to stdout.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_cloud::voxel::voxel_downsample;
use tpt_percept_features::normal::estimate_normals;
use tpt_percept_io::{
    read_kitti_bin, read_pcd, read_ply, write_kitti_to, write_pcd_to, write_ply_to, write_tum_to,
    Encoding, IoError, LoadedCloud, TumPose,
};
use tpt_percept_register::icp::{icp, IcpParams, IcpVariant};
use tpt_percept_slam::lidar_odometry::{LidarOdometry, LidarOdometryParams};

const USAGE: &str = "usage:
  tpt-percept convert    <in> <out> [--ascii]
  tpt-percept downsample <in> <out> --voxel <metres>
  tpt-percept register   <source> <target> [--point-to-point]
  tpt-percept odometry   <scan-dir> [--out <trajectory.tum>]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let (command, rest) = args.split_first().ok_or_else(|| USAGE.to_string())?;
    match command.as_str() {
        "convert" => cmd_convert(rest),
        "downsample" => cmd_downsample(rest),
        "register" => cmd_register(rest),
        "odometry" => cmd_odometry(rest),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

/// Splits `args` into positional arguments and a set of `--flag` values.
struct Parsed {
    positional: Vec<String>,
    ascii: bool,
    point_to_point: bool,
    voxel: Option<f64>,
    out: Option<PathBuf>,
}

fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut p = Parsed {
        positional: Vec::new(),
        ascii: false,
        point_to_point: false,
        voxel: None,
        out: None,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--ascii" => p.ascii = true,
            "--point-to-point" => p.point_to_point = true,
            "--voxel" => {
                let v = it.next().ok_or("--voxel needs a value")?;
                p.voxel = Some(v.parse().map_err(|_| format!("bad --voxel {v:?}"))?);
            }
            "--out" => {
                let v = it.next().ok_or("--out needs a path")?;
                p.out = Some(PathBuf::from(v));
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => p.positional.push(arg.clone()),
        }
    }
    Ok(p)
}

fn need<'a>(p: &'a Parsed, n: usize, what: &str) -> Result<&'a [String], String> {
    if p.positional.len() == n {
        Ok(&p.positional)
    } else {
        Err(format!("expected {n} argument(s): {what}\n\n{USAGE}"))
    }
}

fn load(path: &Path) -> Result<LoadedCloud, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let loaded: Result<LoadedCloud, IoError> = match ext.as_str() {
        "pcd" => read_pcd(path),
        "ply" => read_ply(path),
        "bin" => read_kitti_bin(path),
        other => return Err(format!("unsupported input extension {other:?}")),
    };
    loaded.map_err(|e| format!("{}: {e}", path.display()))
}

fn save(path: &Path, data: &LoadedCloud, ascii: bool) -> Result<(), String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let encoding = if ascii {
        Encoding::Ascii
    } else {
        Encoding::Binary
    };
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let written = match ext.as_str() {
        "pcd" => write_pcd_to(file, data, encoding),
        "ply" => write_ply_to(file, data, encoding),
        "bin" => write_kitti_to(file, data),
        other => return Err(format!("unsupported output extension {other:?}")),
    };
    written.map_err(|e| format!("{}: {e}", path.display()))
}

fn cmd_convert(args: &[String]) -> Result<(), String> {
    let p = parse(args)?;
    let [input, output] = need(&p, 2, "<in> <out>")? else {
        unreachable!("need() checked the length");
    };
    let data = load(Path::new(input))?;
    save(Path::new(output), &data, p.ascii)?;
    println!("wrote {} points to {output}", data.cloud.len());
    Ok(())
}

fn cmd_downsample(args: &[String]) -> Result<(), String> {
    let p = parse(args)?;
    let voxel = p.voxel.ok_or("downsample needs --voxel <metres>")?;
    let [input, output] = need(&p, 2, "<in> <out> --voxel <m>")? else {
        unreachable!("need() checked the length");
    };
    let data = load(Path::new(input))?;
    let reduced =
        voxel_downsample(&data.cloud, voxel).ok_or("voxel size must be finite and > 0")?;
    // Downsampling averages positions, so per-point intensity no longer
    // corresponds to an output point and is dropped.
    let out = LoadedCloud {
        cloud: reduced,
        intensity: None,
    };
    save(Path::new(output), &out, p.ascii)?;
    println!(
        "{} → {} points (voxel {voxel} m)",
        data.cloud.len(),
        out.cloud.len()
    );
    Ok(())
}

fn cmd_register(args: &[String]) -> Result<(), String> {
    let p = parse(args)?;
    let [source_path, target_path] = need(&p, 2, "<source> <target>")? else {
        unreachable!("need() checked the length");
    };
    let source = load(Path::new(source_path))?.cloud;
    let target = load(Path::new(target_path))?.cloud;

    let variant = if p.point_to_point {
        IcpVariant::PointToPoint
    } else {
        IcpVariant::PointToPlane
    };
    let normals: Option<Vec<[f64; 3]>> = match variant {
        IcpVariant::PointToPoint => None,
        IcpVariant::PointToPlane => Some(
            estimate_normals(&target, 10)
                .ok_or("target too small to estimate normals")?
                .iter()
                .map(|n| n.normal)
                .collect(),
        ),
    };
    let params = IcpParams::auto(&source, &target).ok_or("empty cloud")?;
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let result = icp(
        &source,
        &target,
        normals.as_deref(),
        variant,
        identity,
        [0.0; 3],
        &params,
    )
    .map_err(|e| e.to_string())?;

    println!("converged:    {}", result.converged);
    println!("iterations:   {}", result.iterations);
    println!("fitness:      {:.4}", result.fitness);
    println!("rmse (m):     {:.6}", result.rmse);
    println!("translation:  {:?}", result.translation);
    for row in result.rotation {
        println!("rotation row: {row:?}");
    }
    Ok(())
}

fn cmd_odometry(args: &[String]) -> Result<(), String> {
    let p = parse(args)?;
    let [dir] = need(&p, 1, "<scan-dir>")? else {
        unreachable!("need() checked the length");
    };
    let mut scans: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "bin"))
        .collect();
    scans.sort();
    if scans.is_empty() {
        return Err(format!("no .bin scans in {dir}"));
    }

    let mut odom = LidarOdometry::new(LidarOdometryParams::default());
    let mut trajectory = Vec::with_capacity(scans.len());
    let mut failures = 0usize;
    for (k, path) in scans.iter().enumerate() {
        let scan: PointCloud = read_kitti_bin(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .cloud;
        match odom.process(&scan) {
            Ok(step) => {
                if !step.converged {
                    failures += 1;
                }
                trajectory.push(TumPose {
                    timestamp: k as f64,
                    translation: step.translation,
                    quaternion: rotation_to_quaternion(&step.rotation),
                });
            }
            Err(e) => {
                eprintln!("skipping {}: {e}", path.display());
                failures += 1;
            }
        }
    }

    println!(
        "{} scans, {} poses, {} not converged or skipped",
        scans.len(),
        trajectory.len(),
        failures
    );
    if let Some(last) = trajectory.last() {
        println!("final translation: {:?}", last.translation);
    }
    if let Some(out) = &p.out {
        let file = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
        write_tum_to(file, &trajectory).map_err(|e| e.to_string())?;
        println!("wrote trajectory to {}", out.display());
    }
    Ok(())
}

/// Rotation matrix to unit quaternion `(qx, qy, qz, qw)` (Shepperd's method).
fn rotation_to_quaternion(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        )
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        (
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        )
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        (
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        )
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        (
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        )
    };
    [x, y, z, w]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_of_identity_and_quarter_turn() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(rotation_to_quaternion(&id), [0.0, 0.0, 0.0, 1.0]);
        // 90° about z: w = cos 45°, z = sin 45°.
        let rz = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let q = rotation_to_quaternion(&rz);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (q[2] - h).abs() < 1e-12 && (q[3] - h).abs() < 1e-12,
            "{q:?}"
        );
    }

    #[test]
    fn quaternion_branch_for_large_x_rotation() {
        // 180° about x: trace = −1, exercises the x-dominant branch.
        let rx = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let q = rotation_to_quaternion(&rx);
        assert!((q[0].abs() - 1.0).abs() < 1e-12, "{q:?}");
        assert!(q[3].abs() < 1e-12);
    }

    #[test]
    fn parse_collects_flags_and_positionals() {
        let args: Vec<String> = ["a.pcd", "--voxel", "0.2", "b.ply", "--ascii"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let p = parse(&args).unwrap();
        assert_eq!(p.positional, vec!["a.pcd", "b.ply"]);
        assert_eq!(p.voxel, Some(0.2));
        assert!(p.ascii);
        assert!(parse(&["--nope".to_string()]).is_err());
    }
}
