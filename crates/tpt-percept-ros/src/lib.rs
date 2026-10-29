//! Conversion between `tpt-perception` clouds and the ROS 2 `PointCloud2`
//! byte layout.
//!
//! A `sensor_msgs/PointCloud2` message is a flat byte buffer plus a list of
//! fields (name, byte offset, datatype) and a per-point stride. This crate
//! reads and writes that layout without depending on ROS, so it can be unit
//! tested anywhere. A node built on `rclrs` converts its message fields with
//! these functions and nothing more; that node is not part of this crate
//! because it needs a ROS 2 installation to build.
//!
//! Only little-endian messages are supported. Big-endian buffers are rejected
//! rather than byte-swapped silently.

use tpt_percept_cloud::cloud::PointCloud;
use tpt_percept_io::{IoError, IoResult, LoadedCloud};

/// The ROS `PointField` datatype codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    /// `INT8` (1).
    Int8,
    /// `UINT8` (2).
    UInt8,
    /// `INT16` (3).
    Int16,
    /// `UINT16` (4).
    UInt16,
    /// `INT32` (5).
    Int32,
    /// `UINT32` (6).
    UInt32,
    /// `FLOAT32` (7).
    Float32,
    /// `FLOAT64` (8).
    Float64,
}

impl DataType {
    /// The numeric code used in `PointField.datatype`.
    pub fn code(self) -> u8 {
        match self {
            DataType::Int8 => 1,
            DataType::UInt8 => 2,
            DataType::Int16 => 3,
            DataType::UInt16 => 4,
            DataType::Int32 => 5,
            DataType::UInt32 => 6,
            DataType::Float32 => 7,
            DataType::Float64 => 8,
        }
    }

    /// Parses a `PointField.datatype` code.
    pub fn from_code(code: u8) -> IoResult<Self> {
        Ok(match code {
            1 => DataType::Int8,
            2 => DataType::UInt8,
            3 => DataType::Int16,
            4 => DataType::UInt16,
            5 => DataType::Int32,
            6 => DataType::UInt32,
            7 => DataType::Float32,
            8 => DataType::Float64,
            other => return Err(IoError::Format(format!("PointField datatype {other}"))),
        })
    }

    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        match self {
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4,
            DataType::Float64 => 8,
        }
    }
}

/// One `sensor_msgs/PointField`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointField {
    /// Field name, such as `x` or `intensity`.
    pub name: String,
    /// Byte offset of the field within a point.
    pub offset: u32,
    /// Element type.
    pub datatype: DataType,
    /// Number of elements (1 for scalar fields).
    pub count: u32,
}

/// The fields of a `PointCloud2` message that the conversion needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// All fields of a point.
    pub fields: Vec<PointField>,
    /// Bytes per point (`point_step`).
    pub point_step: u32,
}

impl Layout {
    /// The canonical layout written by [`to_point_cloud2`]: `x y z` as
    /// `f32`, followed by `intensity` as `f32` when `with_intensity` is set.
    pub fn xyz(with_intensity: bool) -> Layout {
        let mut fields = vec![
            PointField {
                name: "x".into(),
                offset: 0,
                datatype: DataType::Float32,
                count: 1,
            },
            PointField {
                name: "y".into(),
                offset: 4,
                datatype: DataType::Float32,
                count: 1,
            },
            PointField {
                name: "z".into(),
                offset: 8,
                datatype: DataType::Float32,
                count: 1,
            },
        ];
        let mut point_step = 12;
        if with_intensity {
            fields.push(PointField {
                name: "intensity".into(),
                offset: 12,
                datatype: DataType::Float32,
                count: 1,
            });
            point_step = 16;
        }
        Layout { fields, point_step }
    }

    fn find(&self, name: &str) -> Option<&PointField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Decodes a `PointCloud2` buffer into a cloud, reading `x`, `y`, `z` and,
/// when present, `intensity`. Values are widened to `f64` and intensity is
/// converted to `f32`.
///
/// `is_bigendian` must be false; big-endian messages return an error.
pub fn from_point_cloud2(
    layout: &Layout,
    data: &[u8],
    is_bigendian: bool,
) -> IoResult<LoadedCloud> {
    if is_bigendian {
        return Err(IoError::Unsupported("big-endian PointCloud2"));
    }
    let step = layout.point_step as usize;
    if step == 0 {
        return Err(IoError::Format("point_step is zero".into()));
    }
    if !data.len().is_multiple_of(step) {
        return Err(IoError::Format(
            "data length is not a multiple of point_step".into(),
        ));
    }
    let x = read_scalar(layout, "x")?;
    let y = read_scalar(layout, "y")?;
    let z = read_scalar(layout, "z")?;
    let intensity = layout.find("intensity").cloned();

    let mut cloud = PointCloud::with_capacity(data.len() / step);
    let mut values = intensity
        .as_ref()
        .map(|_| Vec::with_capacity(data.len() / step));
    for point in data.chunks_exact(step) {
        let px = get(point, x)?;
        let py = get(point, y)?;
        let pz = get(point, z)?;
        cloud.push([px, py, pz]);
        if let (Some(list), Some(field)) = (values.as_mut(), intensity.as_ref()) {
            list.push(get(point, field)? as f32);
        }
    }
    Ok(LoadedCloud {
        cloud,
        intensity: values,
    })
}

/// Encodes a cloud into the canonical layout of [`Layout::xyz`]. Returns the
/// layout, the byte buffer, and the point count.
pub fn to_point_cloud2(data: &LoadedCloud) -> (Layout, Vec<u8>) {
    let with_intensity = data.intensity.is_some();
    let layout = Layout::xyz(with_intensity);
    let mut bytes = Vec::with_capacity(data.cloud.len() * layout.point_step as usize);
    for (i, p) in data.cloud.points().iter().enumerate() {
        for c in p {
            bytes.extend_from_slice(&(*c as f32).to_le_bytes());
        }
        if let Some(list) = &data.intensity {
            bytes.extend_from_slice(&list[i].to_le_bytes());
        }
    }
    (layout, bytes)
}

fn read_scalar<'a>(layout: &'a Layout, name: &str) -> IoResult<&'a PointField> {
    let field = layout
        .find(name)
        .ok_or_else(|| IoError::Format(format!("PointCloud2 missing field {name}")))?;
    if field.count != 1 {
        return Err(IoError::Unsupported("PointField with count > 1"));
    }
    Ok(field)
}

fn get(point: &[u8], field: &PointField) -> IoResult<f64> {
    let start = field.offset as usize;
    let end = start + field.datatype.size();
    let bytes = point
        .get(start..end)
        .ok_or_else(|| IoError::Format("field lies outside the point".into()))?;
    Ok(match field.datatype {
        DataType::Float32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::Float64 => f64::from_le_bytes(bytes.try_into().unwrap()),
        DataType::Int8 => bytes[0] as i8 as f64,
        DataType::UInt8 => bytes[0] as f64,
        DataType::Int16 => i16::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::UInt16 => u16::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::Int32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::UInt32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(with_intensity: bool) -> LoadedCloud {
        LoadedCloud {
            cloud: PointCloud::from_points(vec![[0.5, -1.25, 2.0], [3.0, 0.0, -0.5]]),
            intensity: with_intensity.then(|| vec![0.25, 0.75]),
        }
    }

    #[test]
    fn round_trip_with_and_without_intensity() {
        for with in [false, true] {
            let data = sample(with);
            let (layout, bytes) = to_point_cloud2(&data);
            assert_eq!(layout.point_step as usize * data.cloud.len(), bytes.len());
            let back = from_point_cloud2(&layout, &bytes, false).unwrap();
            assert_eq!(back, data);
        }
    }

    #[test]
    fn reads_fields_at_arbitrary_offsets() {
        // A layout with padding: x at 4, y at 12 as f64, z at 20 as i16,
        // point_step 24.
        let layout = Layout {
            fields: vec![
                PointField {
                    name: "x".into(),
                    offset: 4,
                    datatype: DataType::Float32,
                    count: 1,
                },
                PointField {
                    name: "y".into(),
                    offset: 12,
                    datatype: DataType::Float64,
                    count: 1,
                },
                PointField {
                    name: "z".into(),
                    offset: 20,
                    datatype: DataType::Int16,
                    count: 1,
                },
            ],
            point_step: 24,
        };
        let mut point = vec![0u8; 24];
        point[4..8].copy_from_slice(&1.5f32.to_le_bytes());
        point[12..20].copy_from_slice(&(-2.25f64).to_le_bytes());
        point[20..22].copy_from_slice(&(-7i16).to_le_bytes());
        let cloud = from_point_cloud2(&layout, &point, false).unwrap();
        assert_eq!(cloud.cloud.points(), &[[1.5, -2.25, -7.0]]);
    }

    #[test]
    fn big_endian_is_rejected() {
        let (layout, bytes) = to_point_cloud2(&sample(false));
        assert!(matches!(
            from_point_cloud2(&layout, &bytes, true),
            Err(IoError::Unsupported(_))
        ));
    }

    #[test]
    fn misaligned_data_is_rejected() {
        let (layout, mut bytes) = to_point_cloud2(&sample(false));
        bytes.pop();
        assert!(matches!(
            from_point_cloud2(&layout, &bytes, false),
            Err(IoError::Format(_))
        ));
    }

    #[test]
    fn datatype_codes_round_trip() {
        for code in 1..=8u8 {
            assert_eq!(DataType::from_code(code).unwrap().code(), code);
        }
        assert!(DataType::from_code(9).is_err());
    }
}
