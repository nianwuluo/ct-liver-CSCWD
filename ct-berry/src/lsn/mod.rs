//! 肝表面结节 (Liver Surface Nodularity) 评分计算.

mod error;
mod points;
mod sample;

use points::RawSurface;
pub use points::SampledCurve;

pub use sample::{SampleSpec, Spacing};

pub use error::CalcError;

/// 拟合 / LSN 计算运行时错误.
pub type CalcResult<T> = Result<T, CalcError>;
