//! `tpt-percept-io` — reading and writing point clouds and trajectories.
//!
//! * **PCD** (PCL): `ascii` and `binary` `DATA` sections, `x y z` plus an
//!   optional `intensity` field. `binary_compressed` is rejected.
//! * **PLY**: `ascii` and `binary_little_endian` vertex elements with `x y z`
//!   and an optional `intensity`.
//! * **KITTI** odometry `.bin`: little-endian `f32` quadruples `x y z
//!   reflectance`.
//! * **TUM** trajectories: `timestamp tx ty tz qx qy qz qw` per line.
//!
//! Each format has a reader over `BufRead` (or a path wrapper) and a writer
//! over `Write`, so the same code works on files, buffers and sockets.
//! Intensity is returned alongside the geometry in [`LoadedCloud`] because
//! [`PointCloud`] stores positions only.
//!
//! This crate uses `std` for file I/O; the algorithm crates stay `no_std`.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use tpt_percept_cloud::cloud::PointCloud;

pub mod presets;
pub mod recording;

/// Errors from reading or writing a file.
#[derive(Debug)]
pub enum IoError {
    /// Underlying I/O failure.
    Io(io::Error),
    /// The file is malformed or inconsistent with its own header.
    Format(String),
    /// A valid file that uses a feature this crate does not support.
    Unsupported(&'static str),
}

impl core::fmt::Display for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "I/O error: {e}"),
            IoError::Format(what) => write!(f, "malformed file: {what}"),
            IoError::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IoError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for IoError {
    fn from(e: io::Error) -> Self {
        IoError::Io(e)
    }
}

/// Result alias for this crate.
pub type IoResult<T> = Result<T, IoError>;

/// A point cloud plus its optional per-point intensity.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadedCloud {
    /// Point positions (metres, as stored in the file).
    pub cloud: PointCloud,
    /// One intensity per point when the file carries it.
    pub intensity: Option<Vec<f32>>,
}

/// On-disk encoding for the formats that support both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Human-readable text.
    Ascii,
    /// Packed little-endian binary.
    Binary,
}

/// One pose in a TUM trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TumPose {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Translation `(x, y, z)` in metres.
    pub translation: [f64; 3],
    /// Unit quaternion `(qx, qy, qz, qw)`.
    pub quaternion: [f64; 4],
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Reads a PCD file from disk.
pub fn read_pcd(path: impl AsRef<Path>) -> IoResult<LoadedCloud> {
    read_pcd_from(BufReader::new(File::open(path)?))
}

/// Reads a PLY file from disk.
pub fn read_ply(path: impl AsRef<Path>) -> IoResult<LoadedCloud> {
    read_ply_from(BufReader::new(File::open(path)?))
}

/// Reads a KITTI `.bin` scan from disk.
pub fn read_kitti_bin(path: impl AsRef<Path>) -> IoResult<LoadedCloud> {
    read_kitti_from(BufReader::new(File::open(path)?))
}

/// Reads a TUM trajectory file from disk.
pub fn read_tum(path: impl AsRef<Path>) -> IoResult<Vec<TumPose>> {
    read_tum_from(BufReader::new(File::open(path)?))
}

// ---------------------------------------------------------------------------
// PCD
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarType {
    Float,
    Signed,
    Unsigned,
}

#[derive(Clone, Debug)]
struct PcdField {
    name: String,
    size: usize,
    kind: ScalarType,
}

/// Reads a PCD stream (`ascii` or `binary` data).
pub fn read_pcd_from<R: BufRead>(mut reader: R) -> IoResult<LoadedCloud> {
    let mut names: Vec<String> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut types: Vec<char> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut points: Option<usize> = None;
    let mut line = String::new();

    let data_kind: String = loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(IoError::Format("PCD header ends before DATA".into()));
        }
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else { continue };
        let rest: Vec<&str> = parts.collect();
        match key {
            "FIELDS" => names = rest.iter().map(|s| s.to_string()).collect(),
            "SIZE" => sizes = parse_list(&rest, "SIZE")?,
            "TYPE" => {
                types = rest
                    .iter()
                    .map(|s| s.chars().next().unwrap_or('?'))
                    .collect()
            }
            "COUNT" => counts = parse_list(&rest, "COUNT")?,
            "POINTS" => points = Some(parse_one(&rest, "POINTS")?),
            "DATA" => break rest.first().copied().unwrap_or("").to_string(),
            _ => {}
        }
    };

    if names.is_empty() || sizes.len() != names.len() || types.len() != names.len() {
        return Err(IoError::Format("PCD FIELDS/SIZE/TYPE disagree".into()));
    }
    if counts.is_empty() {
        counts = vec![1; names.len()];
    }
    if counts.iter().any(|&c| c != 1) {
        return Err(IoError::Unsupported("PCD fields with COUNT > 1"));
    }
    let fields: Vec<PcdField> = names
        .iter()
        .zip(&sizes)
        .zip(&types)
        .map(|((name, &size), &t)| {
            let kind = match t {
                'F' => ScalarType::Float,
                'I' => ScalarType::Signed,
                'U' => ScalarType::Unsigned,
                _ => return Err(IoError::Format(format!("PCD TYPE {t}"))),
            };
            Ok(PcdField {
                name: name.clone(),
                size,
                kind,
            })
        })
        .collect::<IoResult<_>>()?;
    let ix = field_index(&fields, "x")?;
    let iy = field_index(&fields, "y")?;
    let iz = field_index(&fields, "z")?;
    let ii = fields.iter().position(|f| f.name == "intensity");
    let n = points.unwrap_or(0);

    let mut out = LoadedCloud {
        cloud: PointCloud::with_capacity(n),
        intensity: ii.map(|_| Vec::with_capacity(n)),
    };

    match data_kind.as_str() {
        "ascii" => {
            for text in reader.lines() {
                let text = text?;
                if text.trim().is_empty() {
                    continue;
                }
                let tokens: Vec<&str> = text.split_whitespace().collect();
                if tokens.len() != fields.len() {
                    return Err(IoError::Format("PCD ascii row has wrong arity".into()));
                }
                let value = |i: usize| parse_f64(tokens[i]);
                out.cloud.push([value(ix)?, value(iy)?, value(iz)?]);
                if let (Some(ii), Some(list)) = (ii, out.intensity.as_mut()) {
                    list.push(value(ii)? as f32);
                }
            }
        }
        "binary" => {
            let stride: usize = fields.iter().map(|f| f.size).sum();
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes)?;
            if bytes.len() < stride * n {
                return Err(IoError::Format(
                    "PCD binary data shorter than POINTS".into(),
                ));
            }
            let offsets = field_offsets(&fields);
            for chunk in bytes.chunks_exact(stride).take(n) {
                let read = |i: usize| decode(&chunk[offsets[i]..], &fields[i]);
                out.cloud.push([read(ix)?, read(iy)?, read(iz)?]);
                if let (Some(ii), Some(list)) = (ii, out.intensity.as_mut()) {
                    list.push(read(ii)? as f32);
                }
            }
        }
        "binary_compressed" => {
            return Err(IoError::Unsupported("PCD binary_compressed"));
        }
        other => return Err(IoError::Format(format!("PCD DATA {other:?}"))),
    }
    Ok(out)
}

/// Writes a PCD stream with `x y z` and, when present, `intensity`.
pub fn write_pcd_to<W: Write>(writer: W, data: &LoadedCloud, encoding: Encoding) -> IoResult<()> {
    let mut w = BufWriter::new(writer);
    let with_i = data.intensity.is_some();
    let n = data.cloud.len();
    let fields = if with_i { "x y z intensity" } else { "x y z" };
    let sizes = if with_i { "4 4 4 4" } else { "4 4 4" };
    let types = if with_i { "F F F F" } else { "F F F" };
    let counts = if with_i { "1 1 1 1" } else { "1 1 1" };
    write!(
        w,
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\n"
    )?;
    match encoding {
        Encoding::Ascii => {
            writeln!(w, "DATA ascii")?;
            for (i, p) in data.cloud.points().iter().enumerate() {
                write!(w, "{} {} {}", p[0] as f32, p[1] as f32, p[2] as f32)?;
                if let Some(list) = &data.intensity {
                    write!(w, " {}", list[i])?;
                }
                writeln!(w)?;
            }
        }
        Encoding::Binary => {
            writeln!(w, "DATA binary")?;
            for (i, p) in data.cloud.points().iter().enumerate() {
                for c in p {
                    w.write_all(&(*c as f32).to_le_bytes())?;
                }
                if let Some(list) = &data.intensity {
                    w.write_all(&list[i].to_le_bytes())?;
                }
            }
        }
    }
    w.flush()?;
    Ok(())
}

fn field_index(fields: &[PcdField], name: &str) -> IoResult<usize> {
    fields
        .iter()
        .position(|f| f.name == name)
        .ok_or_else(|| IoError::Format(format!("PCD missing field {name}")))
}

fn field_offsets(fields: &[PcdField]) -> Vec<usize> {
    let mut offset = 0;
    fields
        .iter()
        .map(|f| {
            let at = offset;
            offset += f.size;
            at
        })
        .collect()
}

fn decode(bytes: &[u8], field: &PcdField) -> IoResult<f64> {
    let short = || IoError::Format("PCD binary point truncated".into());
    Ok(match (field.kind, field.size) {
        (ScalarType::Float, 4) => {
            f32::from_le_bytes(bytes.get(..4).ok_or_else(short)?.try_into().unwrap()) as f64
        }
        (ScalarType::Float, 8) => {
            f64::from_le_bytes(bytes.get(..8).ok_or_else(short)?.try_into().unwrap())
        }
        (ScalarType::Signed, 1) => *bytes.first().ok_or_else(short)? as i8 as f64,
        (ScalarType::Unsigned, 1) => *bytes.first().ok_or_else(short)? as f64,
        (ScalarType::Signed, 2) => {
            i16::from_le_bytes(bytes.get(..2).ok_or_else(short)?.try_into().unwrap()) as f64
        }
        (ScalarType::Unsigned, 2) => {
            u16::from_le_bytes(bytes.get(..2).ok_or_else(short)?.try_into().unwrap()) as f64
        }
        (ScalarType::Signed, 4) => {
            i32::from_le_bytes(bytes.get(..4).ok_or_else(short)?.try_into().unwrap()) as f64
        }
        (ScalarType::Unsigned, 4) => {
            u32::from_le_bytes(bytes.get(..4).ok_or_else(short)?.try_into().unwrap()) as f64
        }
        _ => return Err(IoError::Unsupported("PCD scalar size/type combination")),
    })
}

fn parse_list(tokens: &[&str], key: &str) -> IoResult<Vec<usize>> {
    tokens
        .iter()
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| IoError::Format(format!("PCD {key} value {t:?}")))
        })
        .collect()
}

fn parse_one(tokens: &[&str], key: &str) -> IoResult<usize> {
    parse_list(tokens, key)?
        .first()
        .copied()
        .ok_or_else(|| IoError::Format(format!("PCD {key} has no value")))
}

fn parse_f64(token: &str) -> IoResult<f64> {
    token
        .parse::<f64>()
        .map_err(|_| IoError::Format(format!("bad number {token:?}")))
}

// ---------------------------------------------------------------------------
// PLY
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
}

impl PlyType {
    fn parse(name: &str) -> IoResult<PlyType> {
        Ok(match name {
            "char" | "int8" => PlyType::Int8,
            "uchar" | "uint8" => PlyType::UInt8,
            "short" | "int16" => PlyType::Int16,
            "ushort" | "uint16" => PlyType::UInt16,
            "int" | "int32" => PlyType::Int32,
            "uint" | "uint32" => PlyType::UInt32,
            "float" | "float32" => PlyType::Float32,
            "double" | "float64" => PlyType::Float64,
            _ => return Err(IoError::Format(format!("PLY type {name}"))),
        })
    }

    fn size(self) -> usize {
        match self {
            PlyType::Int8 | PlyType::UInt8 => 1,
            PlyType::Int16 | PlyType::UInt16 => 2,
            PlyType::Int32 | PlyType::UInt32 | PlyType::Float32 => 4,
            PlyType::Float64 => 8,
        }
    }

    fn decode_le(self, bytes: &[u8]) -> IoResult<f64> {
        let short = || IoError::Format("PLY binary vertex truncated".into());
        let take = |n: usize| bytes.get(..n).ok_or_else(short);
        Ok(match self {
            PlyType::Int8 => take(1)?[0] as i8 as f64,
            PlyType::UInt8 => take(1)?[0] as f64,
            PlyType::Int16 => i16::from_le_bytes(take(2)?.try_into().unwrap()) as f64,
            PlyType::UInt16 => u16::from_le_bytes(take(2)?.try_into().unwrap()) as f64,
            PlyType::Int32 => i32::from_le_bytes(take(4)?.try_into().unwrap()) as f64,
            PlyType::UInt32 => u32::from_le_bytes(take(4)?.try_into().unwrap()) as f64,
            PlyType::Float32 => f32::from_le_bytes(take(4)?.try_into().unwrap()) as f64,
            PlyType::Float64 => f64::from_le_bytes(take(8)?.try_into().unwrap()),
        })
    }
}

/// Reads the vertex element of a PLY stream.
pub fn read_ply_from<R: BufRead>(mut reader: R) -> IoResult<LoadedCloud> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(IoError::Format("missing ply magic".into()));
    }

    let mut format = String::new();
    let mut vertex_count: Option<usize> = None;
    let mut props: Vec<(String, PlyType)> = Vec::new();
    let mut in_vertex = false;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(IoError::Format("PLY header not terminated".into()));
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["end_header"] => break,
            ["format", kind, ..] => format = kind.to_string(),
            ["element", "vertex", n] => {
                vertex_count = Some(
                    n.parse()
                        .map_err(|_| IoError::Format("PLY vertex count".into()))?,
                );
                in_vertex = true;
            }
            ["element", ..] => in_vertex = false,
            ["property", "list", ..] if in_vertex => {
                return Err(IoError::Unsupported("PLY list property on vertex"));
            }
            ["property", ty, name] if in_vertex => {
                props.push((name.to_string(), PlyType::parse(ty)?));
            }
            _ => {}
        }
    }

    let n = vertex_count.ok_or_else(|| IoError::Format("PLY has no vertex element".into()))?;
    let index = |name: &str| props.iter().position(|(p, _)| p == name);
    let ix = index("x").ok_or_else(|| IoError::Format("PLY missing x".into()))?;
    let iy = index("y").ok_or_else(|| IoError::Format("PLY missing y".into()))?;
    let iz = index("z").ok_or_else(|| IoError::Format("PLY missing z".into()))?;
    let ii = index("intensity");

    let mut out = LoadedCloud {
        cloud: PointCloud::with_capacity(n),
        intensity: ii.map(|_| Vec::with_capacity(n)),
    };

    match format.as_str() {
        "ascii" => {
            let mut rows = reader.lines();
            for _ in 0..n {
                let text = rows
                    .next()
                    .ok_or_else(|| IoError::Format("PLY ascii ends early".into()))??;
                let tokens: Vec<&str> = text.split_whitespace().collect();
                if tokens.len() != props.len() {
                    return Err(IoError::Format("PLY ascii row has wrong arity".into()));
                }
                out.cloud.push([
                    parse_f64(tokens[ix])?,
                    parse_f64(tokens[iy])?,
                    parse_f64(tokens[iz])?,
                ]);
                if let (Some(ii), Some(list)) = (ii, out.intensity.as_mut()) {
                    list.push(parse_f64(tokens[ii])? as f32);
                }
            }
        }
        "binary_little_endian" => {
            let mut offsets = Vec::with_capacity(props.len());
            let mut stride = 0;
            for (_, t) in &props {
                offsets.push(stride);
                stride += t.size();
            }
            let mut bytes = vec![0u8; stride * n];
            reader.read_exact(&mut bytes)?;
            let read = |row: &[u8], i: usize| props[i].1.decode_le(&row[offsets[i]..]);
            for row in bytes.chunks_exact(stride.max(1)).take(n) {
                out.cloud
                    .push([read(row, ix)?, read(row, iy)?, read(row, iz)?]);
                if let (Some(ii), Some(list)) = (ii, out.intensity.as_mut()) {
                    list.push(read(row, ii)? as f32);
                }
            }
        }
        "binary_big_endian" => return Err(IoError::Unsupported("PLY big-endian")),
        other => return Err(IoError::Format(format!("PLY format {other:?}"))),
    }
    Ok(out)
}

/// Writes a PLY stream with `x y z` and, when present, `intensity`.
pub fn write_ply_to<W: Write>(writer: W, data: &LoadedCloud, encoding: Encoding) -> IoResult<()> {
    let mut w = BufWriter::new(writer);
    let format = match encoding {
        Encoding::Ascii => "ascii",
        Encoding::Binary => "binary_little_endian",
    };
    writeln!(w, "ply\nformat {format} 1.0")?;
    writeln!(w, "element vertex {}", data.cloud.len())?;
    writeln!(w, "property float x\nproperty float y\nproperty float z")?;
    if data.intensity.is_some() {
        writeln!(w, "property float intensity")?;
    }
    writeln!(w, "end_header")?;
    for (i, p) in data.cloud.points().iter().enumerate() {
        let intensity = data.intensity.as_ref().map(|list| list[i]);
        match encoding {
            Encoding::Ascii => {
                write!(w, "{} {} {}", p[0] as f32, p[1] as f32, p[2] as f32)?;
                if let Some(v) = intensity {
                    write!(w, " {v}")?;
                }
                writeln!(w)?;
            }
            Encoding::Binary => {
                for c in p {
                    w.write_all(&(*c as f32).to_le_bytes())?;
                }
                if let Some(v) = intensity {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
        }
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// KITTI
// ---------------------------------------------------------------------------

/// Reads a KITTI odometry scan: little-endian `f32` `x y z reflectance`.
pub fn read_kitti_from<R: Read>(mut reader: R) -> IoResult<LoadedCloud> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if bytes.len() % 16 != 0 {
        return Err(IoError::Format(
            "KITTI length is not a multiple of 16".into(),
        ));
    }
    let n = bytes.len() / 16;
    let mut out = LoadedCloud {
        cloud: PointCloud::with_capacity(n),
        intensity: Some(Vec::with_capacity(n)),
    };
    let f = |b: &[u8]| f32::from_le_bytes(b.try_into().unwrap());
    for row in bytes.chunks(16) {
        out.cloud.push([
            f(&row[0..4]) as f64,
            f(&row[4..8]) as f64,
            f(&row[8..12]) as f64,
        ]);
        if let Some(list) = out.intensity.as_mut() {
            list.push(f(&row[12..16]));
        }
    }
    Ok(out)
}

/// Writes a KITTI odometry scan. Missing intensity is written as zero.
pub fn write_kitti_to<W: Write>(writer: W, data: &LoadedCloud) -> IoResult<()> {
    let mut w = BufWriter::new(writer);
    for (i, p) in data.cloud.points().iter().enumerate() {
        for c in p {
            w.write_all(&(*c as f32).to_le_bytes())?;
        }
        let r = data.intensity.as_ref().map_or(0.0, |list| list[i]);
        w.write_all(&r.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// TUM
// ---------------------------------------------------------------------------

/// Reads a TUM trajectory. Blank lines and `#` comments are skipped.
pub fn read_tum_from<R: BufRead>(reader: R) -> IoResult<Vec<TumPose>> {
    let mut poses = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = text
            .split_whitespace()
            .map(parse_f64)
            .collect::<IoResult<_>>()?;
        if v.len() != 8 {
            return Err(IoError::Format(format!(
                "TUM row has {} fields, want 8",
                v.len()
            )));
        }
        poses.push(TumPose {
            timestamp: v[0],
            translation: [v[1], v[2], v[3]],
            quaternion: [v[4], v[5], v[6], v[7]],
        });
    }
    Ok(poses)
}

/// Writes a TUM trajectory.
pub fn write_tum_to<W: Write>(writer: W, poses: &[TumPose]) -> IoResult<()> {
    let mut w = BufWriter::new(writer);
    for p in poses {
        writeln!(
            w,
            "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            p.timestamp,
            p.translation[0],
            p.translation[1],
            p.translation[2],
            p.quaternion[0],
            p.quaternion[1],
            p.quaternion[2],
            p.quaternion[3],
        )?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample(with_intensity: bool) -> LoadedCloud {
        // Values chosen to be exact in f32 so round trips compare equal.
        let points = vec![[0.5, -1.25, 2.0], [3.0, 0.0, -0.5], [-2.5, 4.0, 0.125]];
        LoadedCloud {
            cloud: PointCloud::from_points(points),
            intensity: with_intensity.then(|| vec![0.25, 0.5, 1.0]),
        }
    }

    #[test]
    fn pcd_round_trips_both_encodings() {
        for &enc in &[Encoding::Ascii, Encoding::Binary] {
            for &i in &[false, true] {
                let data = sample(i);
                let mut buf = Vec::new();
                write_pcd_to(&mut buf, &data, enc).unwrap();
                let back = read_pcd_from(Cursor::new(buf)).unwrap();
                assert_eq!(back, data, "{enc:?} intensity={i}");
            }
        }
    }

    #[test]
    fn ply_round_trips_both_encodings() {
        for &enc in &[Encoding::Ascii, Encoding::Binary] {
            for &i in &[false, true] {
                let data = sample(i);
                let mut buf = Vec::new();
                write_ply_to(&mut buf, &data, enc).unwrap();
                let back = read_ply_from(Cursor::new(buf)).unwrap();
                assert_eq!(back, data, "{enc:?} intensity={i}");
            }
        }
    }

    #[test]
    fn kitti_round_trips() {
        let data = sample(true);
        let mut buf = Vec::new();
        write_kitti_to(&mut buf, &data).unwrap();
        assert_eq!(buf.len(), 16 * 3);
        let back = read_kitti_from(Cursor::new(buf)).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn kitti_rejects_truncated_input() {
        assert!(matches!(
            read_kitti_from(Cursor::new(vec![0u8; 17])),
            Err(IoError::Format(_))
        ));
    }

    #[test]
    fn tum_round_trips_and_skips_comments() {
        let poses = vec![TumPose {
            timestamp: 1.5,
            translation: [1.0, 2.0, 3.0],
            quaternion: [0.0, 0.0, 0.0, 1.0],
        }];
        let mut buf = Vec::new();
        write_tum_to(&mut buf, &poses).unwrap();
        let mut text = b"# header\n\n".to_vec();
        text.extend(buf);
        let back = read_tum_from(Cursor::new(text)).unwrap();
        assert_eq!(back, poses);
    }

    #[test]
    fn pcd_rejects_compressed_data() {
        let text = "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 0\nDATA binary_compressed\n";
        assert!(matches!(
            read_pcd_from(Cursor::new(text.as_bytes())),
            Err(IoError::Unsupported(_))
        ));
    }

    #[test]
    fn ply_rejects_list_properties() {
        let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty list uchar int vertex_indices\nend_header\n";
        assert!(read_ply_from(Cursor::new(text.as_bytes())).is_err());
    }
}
