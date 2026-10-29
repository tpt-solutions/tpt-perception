//! Estimates the fixed transform between two sensors from motion pairs.
//!
//! Each input line is one synchronised epoch with twelve comma-separated
//! values: sensor A's axis-angle rotation and translation, then sensor B's.
//! Lines starting with `#` are ignored.
//!
//! Usage: `{{project-name}} <motion-pairs.csv>`

use tpt_perception::fusion::calibration::{calibrate_extrinsic, MotionPair};

fn parse_line(line: &str) -> Result<MotionPair, String> {
    let v: Vec<f64> = line
        .split(',')
        .map(|t| t.trim().parse::<f64>().map_err(|e| format!("{t:?}: {e}")))
        .collect::<Result<_, _>>()?;
    if v.len() != 12 {
        return Err(format!("expected 12 values, found {}", v.len()));
    }
    Ok(MotionPair {
        rot_a: [v[0], v[1], v[2]],
        trans_a: [v[3], v[4], v[5]],
        rot_b: [v[6], v[7], v[8]],
        trans_b: [v[9], v[10], v[11]],
    })
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: {{project-name}} <motion-pairs.csv>");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(&path).expect("readable CSV");
    let mut pairs = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_line(line) {
            Ok(p) => pairs.push(p),
            Err(e) => {
                eprintln!("line {}: {e}", n + 1);
                std::process::exit(1);
            }
        }
    }

    match calibrate_extrinsic(&pairs) {
        Ok(ext) => {
            println!("pairs used: {}", pairs.len());
            println!("rotation:");
            for row in ext.rotation {
                println!("  {row:?}");
            }
            println!("translation (m): {:?}", ext.translation);
        }
        Err(e) => {
            eprintln!("calibration failed: {e}");
            std::process::exit(1);
        }
    }
}
