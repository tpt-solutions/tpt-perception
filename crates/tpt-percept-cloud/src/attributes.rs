//! Per-point attributes carried alongside a [`PointCloud`].
//!
//! [`PointCloud`] stores positions only, which keeps every algorithm simple.
//! Sensors add data per point (reflectance, laser ring, capture time), and
//! that data must stay aligned with the points through selection and
//! transforms. [`AttributedCloud`] keeps each optional attribute column the
//! same length as the cloud and applies every operation to all of them.

use alloc::vec::Vec;

use crate::cloud::PointCloud;

/// Error when attribute columns do not match the cloud length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeLengthError {
    /// Name of the column with the wrong length.
    pub column: &'static str,
    /// Expected length (the number of points).
    pub expected: usize,
    /// Length actually supplied.
    pub found: usize,
}

impl core::fmt::Display for AttributeLengthError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "attribute `{}` has {} entries for {} points",
            self.column, self.found, self.expected
        )
    }
}

impl core::error::Error for AttributeLengthError {}

/// A point cloud with optional per-point columns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttributedCloud {
    cloud: PointCloud,
    intensity: Option<Vec<f32>>,
    timestamp: Option<Vec<f64>>,
    ring: Option<Vec<u16>>,
}

impl AttributedCloud {
    /// A cloud with no attribute columns.
    pub fn new(cloud: PointCloud) -> Self {
        AttributedCloud {
            cloud,
            intensity: None,
            timestamp: None,
            ring: None,
        }
    }

    /// Attaches the intensity column.
    pub fn with_intensity(mut self, values: Vec<f32>) -> Result<Self, AttributeLengthError> {
        check(self.cloud.len(), "intensity", values.len())?;
        self.intensity = Some(values);
        Ok(self)
    }

    /// Attaches per-point capture timestamps (seconds, or a normalised scan
    /// fraction, as the caller prefers).
    pub fn with_timestamp(mut self, values: Vec<f64>) -> Result<Self, AttributeLengthError> {
        check(self.cloud.len(), "timestamp", values.len())?;
        self.timestamp = Some(values);
        Ok(self)
    }

    /// Attaches the laser ring index of each point.
    pub fn with_ring(mut self, values: Vec<u16>) -> Result<Self, AttributeLengthError> {
        check(self.cloud.len(), "ring", values.len())?;
        self.ring = Some(values);
        Ok(self)
    }

    /// Borrows the positions.
    pub fn cloud(&self) -> &PointCloud {
        &self.cloud
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.cloud.len()
    }

    /// True if the cloud has no points.
    pub fn is_empty(&self) -> bool {
        self.cloud.is_empty()
    }

    /// Intensity column, if present.
    pub fn intensity(&self) -> Option<&[f32]> {
        self.intensity.as_deref()
    }

    /// Timestamp column, if present.
    pub fn timestamp(&self) -> Option<&[f64]> {
        self.timestamp.as_deref()
    }

    /// Ring column, if present.
    pub fn ring(&self) -> Option<&[u16]> {
        self.ring.as_deref()
    }

    /// Keeps the listed points, in the order given, with every column.
    /// Out-of-range indices are ignored, as in [`PointCloud::select`].
    pub fn select(&self, indices: &[usize]) -> AttributedCloud {
        AttributedCloud {
            cloud: self.cloud.select(indices),
            intensity: pick(&self.intensity, indices),
            timestamp: pick(&self.timestamp, indices),
            ring: pick(&self.ring, indices),
        }
    }

    /// Applies a rigid transform to the positions. Attributes are unchanged,
    /// since rigid motion does not alter a point's reflectance, ring or time.
    pub fn transformed_rigid(
        &self,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) -> AttributedCloud {
        AttributedCloud {
            cloud: self.cloud.transformed_rigid(rotation, translation),
            intensity: self.intensity.clone(),
            timestamp: self.timestamp.clone(),
            ring: self.ring.clone(),
        }
    }

    /// Appends another attributed cloud. Both sides must have the same
    /// attribute columns, or the result drops any column that is missing on
    /// either side.
    pub fn extend_from(&mut self, other: &AttributedCloud) {
        fn merge<T: Clone>(mine: &mut Option<Vec<T>>, theirs: &Option<Vec<T>>) {
            match (mine.as_mut(), theirs) {
                (Some(a), Some(b)) => a.extend_from_slice(b),
                _ => *mine = None,
            }
        }
        merge(&mut self.intensity, &other.intensity);
        merge(&mut self.timestamp, &other.timestamp);
        merge(&mut self.ring, &other.ring);
        self.cloud.extend_from(&other.cloud);
    }
}

/// Gathers `indices` from an optional column, skipping out-of-range entries.
fn pick<T: Clone>(column: &Option<Vec<T>>, indices: &[usize]) -> Option<Vec<T>> {
    column
        .as_ref()
        .map(|v| indices.iter().filter_map(|&i| v.get(i).cloned()).collect())
}

fn check(expected: usize, column: &'static str, found: usize) -> Result<(), AttributeLengthError> {
    if expected == found {
        Ok(())
    } else {
        Err(AttributeLengthError {
            column,
            expected,
            found,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample() -> AttributedCloud {
        AttributedCloud::new(PointCloud::from_points(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]))
        .with_intensity(vec![0.1, 0.2, 0.3])
        .unwrap()
        .with_timestamp(vec![0.0, 0.5, 1.0])
        .unwrap()
        .with_ring(vec![3, 4, 5])
        .unwrap()
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let cloud = PointCloud::from_points(vec![[0.0; 3]]);
        let err = AttributedCloud::new(cloud)
            .with_ring(vec![1, 2])
            .unwrap_err();
        assert_eq!(err.column, "ring");
        assert_eq!(err.expected, 1);
        assert_eq!(err.found, 2);
    }

    #[test]
    fn select_keeps_columns_aligned() {
        let picked = sample().select(&[2, 0]);
        assert_eq!(picked.cloud().points(), &[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert_eq!(picked.intensity(), Some(&[0.3, 0.1][..]));
        assert_eq!(picked.timestamp(), Some(&[1.0, 0.0][..]));
        assert_eq!(picked.ring(), Some(&[5, 3][..]));
    }

    #[test]
    fn transform_leaves_attributes_untouched() {
        let moved = sample().transformed_rigid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        );
        assert_eq!(moved.cloud().points()[0], [1.0, 0.0, 0.0]);
        assert_eq!(moved.ring(), sample().ring());
    }

    #[test]
    fn extend_drops_columns_missing_on_either_side() {
        let mut a = sample();
        let b = AttributedCloud::new(PointCloud::from_points(vec![[9.0; 3]]));
        a.extend_from(&b);
        assert_eq!(a.len(), 4);
        assert!(a.intensity().is_none());
        assert!(a.ring().is_none());
    }
}
