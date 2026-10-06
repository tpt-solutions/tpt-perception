//! Errors produced by the mapping crate.

/// Map errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapError {
    /// A parameter or input violated its documented constraint.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for MapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MapError::InvalidParameter(what) => write!(f, "invalid map parameter: {what}"),
        }
    }
}

impl core::error::Error for MapError {}
