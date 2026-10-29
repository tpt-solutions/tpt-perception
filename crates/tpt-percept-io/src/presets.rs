//! Nominal specifications of common spinning LiDARs.
//!
//! These are starting points for configuring odometry and registration, not
//! calibrated values. Figures are from manufacturer datasheets as commonly
//! quoted; confirm them for your unit. `suggested_gate_m` is a heuristic
//! correspondence gate for ICP on a well-initialised pose.

/// Nominal parameters for one sensor model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensorPreset {
    /// Human-readable model name.
    pub name: &'static str,
    /// Number of laser channels (rings).
    pub rings: u16,
    /// Nominal maximum range (metres).
    pub max_range_m: f64,
    /// Vertical field of view `(min, max)` in degrees.
    pub vertical_fov_deg: (f64, f64),
    /// Rotation rate (Hz). Frames per second for a full revolution.
    pub rate_hz: f64,
    /// Suggested ICP correspondence gate (metres) for consecutive scans.
    pub suggested_gate_m: f64,
}

/// Velodyne HDL-64E (also the sensor behind the KITTI odometry dataset).
pub const VELODYNE_HDL64E: SensorPreset = SensorPreset {
    name: "Velodyne HDL-64E",
    rings: 64,
    max_range_m: 120.0,
    vertical_fov_deg: (-24.9, 2.0),
    rate_hz: 10.0,
    suggested_gate_m: 1.0,
};

/// Velodyne VLP-16 (Puck).
pub const VELODYNE_VLP16: SensorPreset = SensorPreset {
    name: "Velodyne VLP-16",
    rings: 16,
    max_range_m: 100.0,
    vertical_fov_deg: (-15.0, 15.0),
    rate_hz: 10.0,
    suggested_gate_m: 0.5,
};

/// Ouster OS1-64.
pub const OUSTER_OS1_64: SensorPreset = SensorPreset {
    name: "Ouster OS1-64",
    rings: 64,
    max_range_m: 120.0,
    vertical_fov_deg: (-22.5, 22.5),
    rate_hz: 10.0,
    suggested_gate_m: 0.5,
};

/// Every preset in this module.
pub const ALL: [SensorPreset; 3] = [VELODYNE_HDL64E, VELODYNE_VLP16, OUSTER_OS1_64];

/// Looks up a preset by a case-insensitive substring of its name, such as
/// `"vlp-16"` or `"os1"`.
pub fn find(query: &str) -> Option<SensorPreset> {
    let normalise = |s: &str| s.to_ascii_lowercase().replace(' ', "");
    let q = normalise(query);
    ALL.iter().copied().find(|p| normalise(p.name).contains(&q))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_internally_consistent() {
        for p in ALL {
            assert!(p.rings > 0);
            assert!(p.max_range_m > 0.0);
            assert!(p.vertical_fov_deg.0 < p.vertical_fov_deg.1);
            assert!(p.suggested_gate_m > 0.0);
        }
    }

    #[test]
    fn find_is_case_and_space_insensitive() {
        assert_eq!(find("vlp-16"), Some(VELODYNE_VLP16));
        assert_eq!(find("Ouster OS1"), Some(OUSTER_OS1_64));
        assert_eq!(find("hdl-64e"), Some(VELODYNE_HDL64E));
        assert_eq!(find("no such sensor"), None);
    }
}
