//! Semantic maps: labeled point clouds for object recognition and scene
//! understanding.
//!
//! A [`SemanticCloud`] couples positions with per-point class labels and
//! confidences, and derives per-class statistics (counts, centroids,
//! bounding boxes) that downstream consumers (detection, navigation)
//! consume. Class identifiers are caller-owned `u16`s so learned or
//! hand-crafted taxonomies plug in without this crate knowing about them.

use alloc::vec::Vec;

use crate::error::MapError;

/// One labeled point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabeledPoint {
    /// Position (metres).
    pub position: [f64; 3],
    /// Class identifier (caller-owned taxonomy).
    pub class: u16,
    /// Detection confidence in [0, 1].
    pub confidence: f32,
}

/// A labeled point cloud.
#[derive(Clone, Debug, Default)]
pub struct SemanticCloud {
    points: Vec<LabeledPoint>,
}

impl SemanticCloud {
    /// An empty cloud.
    pub fn new() -> Self {
        SemanticCloud { points: Vec::new() }
    }

    /// Builds from labeled points, validating confidences.
    pub fn from_points(points: Vec<LabeledPoint>) -> Result<Self, MapError> {
        for p in &points {
            if !(p.confidence.is_finite() && (0.0..=1.0).contains(&p.confidence)) {
                return Err(MapError::InvalidParameter("confidence must be in [0, 1]"));
            }
            if p.position.iter().any(|v| !v.is_finite()) {
                return Err(MapError::InvalidParameter("non-finite position"));
            }
        }
        Ok(SemanticCloud { points })
    }

    /// Point count.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// All labeled points.
    pub fn points(&self) -> &[LabeledPoint] {
        &self.points
    }

    /// Appends one point (validated).
    pub fn push(&mut self, point: LabeledPoint) -> Result<(), MapError> {
        if !(point.confidence.is_finite() && (0.0..=1.0).contains(&point.confidence)) {
            return Err(MapError::InvalidParameter("confidence must be in [0, 1]"));
        }
        if point.position.iter().any(|v| !v.is_finite()) {
            return Err(MapError::InvalidParameter("non-finite position"));
        }
        self.points.push(point);
        Ok(())
    }

    /// Keeps only points of the given class.
    pub fn filter_class(&self, class: u16) -> SemanticCloud {
        SemanticCloud {
            points: self
                .points
                .iter()
                .copied()
                .filter(|p| p.class == class)
                .collect(),
        }
    }

    /// Keeps only points at or above a confidence threshold.
    pub fn filter_confidence(&self, threshold: f32) -> SemanticCloud {
        SemanticCloud {
            points: self
                .points
                .iter()
                .copied()
                .filter(|p| p.confidence >= threshold)
                .collect(),
        }
    }

    /// Per-class statistics, ordered by class id.
    pub fn class_statistics(&self) -> Vec<ClassStatistics> {
        let mut order: Vec<u16> = Vec::new();
        for p in &self.points {
            if !order.contains(&p.class) {
                order.push(p.class);
            }
        }
        order.sort_unstable();
        order
            .into_iter()
            .filter_map(|class| self.class_statistics_of(class))
            .collect()
    }

    /// Statistics for one class.
    pub fn class_statistics_of(&self, class: u16) -> Option<ClassStatistics> {
        let mut count = 0usize;
        let mut centroid = [0.0; 3];
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut confidence_sum = 0.0f32;
        for p in &self.points {
            if p.class != class {
                continue;
            }
            count += 1;
            for i in 0..3 {
                centroid[i] += p.position[i];
                min[i] = min[i].min(p.position[i]);
                max[i] = max[i].max(p.position[i]);
            }
            confidence_sum += p.confidence;
        }
        if count == 0 {
            return None;
        }
        let n = count as f64;
        Some(ClassStatistics {
            class,
            count,
            centroid: [centroid[0] / n, centroid[1] / n, centroid[2] / n],
            min,
            max,
            mean_confidence: confidence_sum / count as f32,
        })
    }
}

/// Aggregate statistics for one semantic class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassStatistics {
    /// Class identifier.
    pub class: u16,
    /// Number of points.
    pub count: usize,
    /// Mean position (metres).
    pub centroid: [f64; 3],
    /// Axis-aligned minimum.
    pub min: [f64; 3],
    /// Axis-aligned maximum.
    pub max: [f64; 3],
    /// Mean confidence.
    pub mean_confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample_cloud() -> SemanticCloud {
        SemanticCloud::from_points(vec![
            LabeledPoint {
                position: [1.0, 0.0, 0.0],
                class: 1,
                confidence: 0.9,
            },
            LabeledPoint {
                position: [1.1, 0.1, 0.0],
                class: 1,
                confidence: 0.8,
            },
            LabeledPoint {
                position: [5.0, 5.0, 1.0],
                class: 2,
                confidence: 0.95,
            },
            LabeledPoint {
                position: [0.0, 0.0, 0.0],
                class: 1,
                confidence: 0.2,
            },
        ])
        .unwrap()
    }

    #[test]
    fn class_statistics_aggregate() {
        let cloud = sample_cloud();
        let stats = cloud.class_statistics();
        assert_eq!(stats.len(), 2);
        let car = stats.iter().find(|s| s.class == 1).unwrap();
        assert_eq!(car.count, 3);
        assert!((car.centroid[0] - 0.7).abs() < 1e-12);
        assert!((car.mean_confidence - (0.9 + 0.8 + 0.2) / 3.0).abs() < 1e-6);
        assert_eq!(car.min, [0.0, 0.0, 0.0]);
        assert_eq!(car.max, [1.1, 0.1, 0.0]);
    }

    #[test]
    fn filters_compose() {
        let cloud = sample_cloud();
        let high_conf_cars = cloud.filter_class(1).filter_confidence(0.5);
        assert_eq!(high_conf_cars.len(), 2);
        assert!(high_conf_cars.points().iter().all(|p| p.confidence >= 0.5));
    }

    #[test]
    fn invalid_confidence_rejected() {
        assert!(SemanticCloud::from_points(vec![LabeledPoint {
            position: [0.0; 3],
            class: 0,
            confidence: 1.5,
        }])
        .is_err());
        let mut cloud = SemanticCloud::new();
        assert!(cloud
            .push(LabeledPoint {
                position: [f64::NAN; 3],
                class: 0,
                confidence: 0.5
            })
            .is_err());
    }

    #[test]
    fn unknown_class_has_no_statistics() {
        let cloud = sample_cloud();
        assert!(cloud.class_statistics_of(99).is_none());
    }
}
