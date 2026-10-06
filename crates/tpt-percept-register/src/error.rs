//! Registration errors shared by the algorithms in this crate.

/// Errors produced by registration algorithms.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegistrationError {
    /// An input cloud was empty.
    EmptyCloud,
    /// A parameter violated its documented constraint.
    InvalidParameter(&'static str),
    /// A required input was not provided (e.g. normals for point-to-plane).
    MissingNormals,
    /// Input dimensions do not agree.
    DimensionMismatch {
        /// Description of the mismatch.
        what: &'static str,
    },
    /// The closed-form alignment inside the loop failed (degenerate
    /// correspondences).
    AlignmentFailed(tpt_percept_core::error::CoreError),
    /// A linear system inside the optimiser was singular.
    LinearSolveFailed(alloc::string::String),
    /// The algorithm could not produce a valid hypothesis.
    NoSolution,
}

impl core::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RegistrationError::EmptyCloud => write!(f, "input cloud is empty"),
            RegistrationError::InvalidParameter(what) => {
                write!(f, "invalid parameter: {what}")
            }
            RegistrationError::MissingNormals => {
                write!(f, "point-to-plane registration requires target normals")
            }
            RegistrationError::DimensionMismatch { what } => {
                write!(f, "dimension mismatch: {what}")
            }
            RegistrationError::AlignmentFailed(e) => write!(f, "alignment failed: {e}"),
            RegistrationError::LinearSolveFailed(what) => {
                write!(f, "linear solve failed: {what}")
            }
            RegistrationError::NoSolution => write!(f, "no registration hypothesis found"),
        }
    }
}

impl core::error::Error for RegistrationError {}
