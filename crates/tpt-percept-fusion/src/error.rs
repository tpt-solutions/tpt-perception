//! Errors produced by the fusion crate.

/// Fusion errors.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FusionError {
    /// A parameter violated its documented constraint.
    InvalidParameter(&'static str),
    /// Not enough input data.
    InsufficientData(&'static str),
    /// Matrix dimensions do not agree.
    DimensionMismatch(&'static str),
    /// A required matrix inverse failed.
    SingularSystem(&'static str),
}

impl core::fmt::Display for FusionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FusionError::InvalidParameter(what) => write!(f, "invalid parameter: {what}"),
            FusionError::InsufficientData(what) => write!(f, "insufficient data: {what}"),
            FusionError::DimensionMismatch(what) => write!(f, "dimension mismatch: {what}"),
            FusionError::SingularSystem(what) => write!(f, "singular system: {what}"),
        }
    }
}

impl core::error::Error for FusionError {}
