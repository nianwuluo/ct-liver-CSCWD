//! 运行时错误.

/// 取样或拟合的运行时错误.
#[derive(Debug, Clone)]
pub enum CalcError {
    /// 肝左外区总长度不足.
    LengthTooShort,

    /// 拟合曲线不存在.
    FitCurveDoesNotExist,

    /// 采样点不足以做实际拟合工作.
    ///
    /// 第一个参数代表目前已有的点, 第二个参数代表实际拟合需要的最少点数.
    TooFewSamples(u32, u32),
}
