//! Error types for `tpt-percept-core`.

use core::fmt;

/// Errors produced by geometric operations in `tpt-percept-core`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoreError {
    /// A coordinate or axis was non-finite (NaN or ±inf), or a zero-length
    /// vector was given where a direction was required.
    InvalidInput {
        /// Human-readable description of the violated input constraint.
        what: &'static str,
    },
    /// The configuration is geometrically degenerate for the requested
    /// operation (e.g. rank-deficient alignment, collinear points).
    Degenerate {
        /// Human-readable description of the degeneracy.
        what: &'static str,
    },
    /// Dimensions of inputs do not match.
    DimensionMismatch {
        /// Human-readable description of the mismatch.
        what: &'static str,
    },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::InvalidInput { what } => write!(f, "invalid input: {what}"),
            CoreError::Degenerate { what } => write!(f, "degenerate configuration: {what}"),
            CoreError::DimensionMismatch { what } => write!(f, "dimension mismatch: {what}"),
        }
    }
}

impl core::error::Error for CoreError {}

/// Convenience alias for results produced by core geometric operations.
pub type CoreResult<T> = Result<T, CoreError>;
