//! SLAM errors.

/// Errors produced by SLAM algorithms.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlamError {
    /// Not enough input data for the requested computation.
    InsufficientData(&'static str),
    /// A parameter violated its documented constraint.
    InvalidParameter(&'static str),
    /// A geometric decomposition failed.
    DecompositionFailed(alloc::string::String),
    /// The algorithm could not produce a valid hypothesis.
    NoSolution,
    /// Registration inside the SLAM pipeline failed.
    Registration(tpt_percept_register::error::RegistrationError),
}

impl core::fmt::Display for SlamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SlamError::InsufficientData(what) => write!(f, "insufficient data: {what}"),
            SlamError::InvalidParameter(what) => write!(f, "invalid parameter: {what}"),
            SlamError::DecompositionFailed(what) => write!(f, "decomposition failed: {what}"),
            SlamError::NoSolution => write!(f, "no solution found"),
            SlamError::Registration(e) => write!(f, "registration failed: {e}"),
        }
    }
}

impl core::error::Error for SlamError {}

impl From<tpt_percept_register::error::RegistrationError> for SlamError {
    fn from(e: tpt_percept_register::error::RegistrationError) -> Self {
        SlamError::Registration(e)
    }
}
