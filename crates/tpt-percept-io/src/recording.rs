//! Bit-exact recordings of scan sequences for deterministic replay.
//!
//! Every algorithm in `tpt-perception` is deterministic given its input (all
//! randomness is seeded), so a recorded sequence replayed through the same
//! parameters reproduces the same outputs bit for bit. This module stores the
//! input frames exactly: positions are written as little-endian `f64`, not
//! rounded to text. Attach the parameters as the free-form `metadata` string
//! so a bug report carries everything needed to reproduce it.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic        8 bytes  b"TPTREC01"
//! meta_len     u32
//! metadata     meta_len bytes, UTF-8
//! frame_count  u32
//! per frame:
//!   point_count  u64
//!   flags        u8     bit 0: intensity column present
//!   points       point_count × 3 × f64
//!   intensity    point_count × f32   (only when flag bit 0 is set)
//! ```

use std::io::{BufWriter, Read, Write};

use crate::{IoError, IoResult, LoadedCloud};
use tpt_percept_cloud::cloud::PointCloud;

const MAGIC: &[u8; 8] = b"TPTREC01";
const FLAG_INTENSITY: u8 = 1;
/// Upper bound on preallocation, so a corrupt count cannot exhaust memory.
const MAX_PREALLOC: usize = 1 << 20;

/// A sequence of frames plus a description of how they were produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recording {
    /// Free-form description, typically the parameters used to process the
    /// frames.
    pub metadata: String,
    /// Frames in capture order.
    pub frames: Vec<LoadedCloud>,
}

/// Writes a recording.
pub fn write_recording_to<W: Write>(writer: W, recording: &Recording) -> IoResult<()> {
    let mut w = BufWriter::new(writer);
    w.write_all(MAGIC)?;
    let meta = recording.metadata.as_bytes();
    w.write_all(&u32_len(meta.len(), "metadata")?.to_le_bytes())?;
    w.write_all(meta)?;
    w.write_all(&u32_len(recording.frames.len(), "frame count")?.to_le_bytes())?;
    for frame in &recording.frames {
        w.write_all(&(frame.cloud.len() as u64).to_le_bytes())?;
        let flags = if frame.intensity.is_some() {
            FLAG_INTENSITY
        } else {
            0
        };
        w.write_all(&[flags])?;
        for p in frame.cloud.points() {
            for c in p {
                w.write_all(&c.to_le_bytes())?;
            }
        }
        if let Some(values) = &frame.intensity {
            if values.len() != frame.cloud.len() {
                return Err(IoError::Format(
                    "intensity length differs from point count".into(),
                ));
            }
            for v in values {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

/// Reads a recording, validating the magic and every length.
pub fn read_recording_from<R: Read>(mut reader: R) -> IoResult<Recording> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(IoError::Format("not a TPTREC01 recording".into()));
    }
    let meta_len = read_u32(&mut reader)? as usize;
    let mut meta = vec![0u8; meta_len];
    reader.read_exact(&mut meta)?;
    let metadata =
        String::from_utf8(meta).map_err(|_| IoError::Format("metadata is not UTF-8".into()))?;

    let frame_count = read_u32(&mut reader)? as usize;
    let mut frames = Vec::with_capacity(frame_count.min(MAX_PREALLOC));
    for _ in 0..frame_count {
        let n = read_u64(&mut reader)?;
        let n = usize::try_from(n).map_err(|_| IoError::Format("point count overflows".into()))?;
        let mut flags = [0u8; 1];
        reader.read_exact(&mut flags)?;
        let mut cloud = PointCloud::with_capacity(n.min(MAX_PREALLOC));
        for _ in 0..n {
            let x = read_f64(&mut reader)?;
            let y = read_f64(&mut reader)?;
            let z = read_f64(&mut reader)?;
            cloud.push([x, y, z]);
        }
        let intensity = if flags[0] & FLAG_INTENSITY != 0 {
            let mut values = Vec::with_capacity(n.min(MAX_PREALLOC));
            for _ in 0..n {
                values.push(read_f32(&mut reader)?);
            }
            Some(values)
        } else {
            None
        };
        frames.push(LoadedCloud { cloud, intensity });
    }
    Ok(Recording { metadata, frames })
}

fn u32_len(n: usize, what: &'static str) -> IoResult<u32> {
    u32::try_from(n).map_err(|_| IoError::Format(format!("{what} exceeds u32::MAX")))
}

fn read_u32<R: Read>(r: &mut R) -> IoResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> IoResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> IoResult<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64<R: Read>(r: &mut R) -> IoResult<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample() -> Recording {
        Recording {
            metadata: "icp gate=0.5 seed=42".into(),
            frames: vec![
                LoadedCloud {
                    cloud: PointCloud::from_points(vec![
                        [0.1, -0.2, 0.3],
                        [1.0 / 3.0, 2.5, -1e-300],
                    ]),
                    intensity: Some(vec![0.5, 0.25]),
                },
                LoadedCloud {
                    cloud: PointCloud::from_points(vec![[9.0, 8.0, 7.0]]),
                    intensity: None,
                },
            ],
        }
    }

    #[test]
    fn round_trip_is_bit_exact() {
        let rec = sample();
        let mut buf = Vec::new();
        write_recording_to(&mut buf, &rec).unwrap();
        let back = read_recording_from(Cursor::new(buf)).unwrap();
        assert_eq!(back, rec);
        // f64 equality above is exact; confirm the awkward value survived.
        assert_eq!(
            back.frames[0].cloud.points()[1][0].to_bits(),
            (1.0f64 / 3.0).to_bits()
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let err = read_recording_from(Cursor::new(b"NOTARECX0000".to_vec())).unwrap_err();
        assert!(matches!(err, IoError::Format(_)));
    }

    #[test]
    fn truncated_recording_is_an_error_not_a_panic() {
        let mut buf = Vec::new();
        write_recording_to(&mut buf, &sample()).unwrap();
        buf.truncate(buf.len() - 5);
        assert!(read_recording_from(Cursor::new(buf)).is_err());
    }

    #[test]
    fn huge_claimed_count_does_not_allocate_eagerly() {
        // Claims 2^40 points in one frame but supplies no data.
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(1u64 << 40).to_le_bytes());
        buf.push(0);
        assert!(read_recording_from(Cursor::new(buf)).is_err());
    }
}
