//! 曲线拟合.
//!
//! 给定一系列点 `(x, y)`, 该模块可以拟合出一条曲线.

use ndarray::ArrayView1;

mod cubic_spline;
mod polynomial;

type VecPair<T> = (Vec<T>, Vec<T>);

/// 曲线类型.
#[derive(Copy, Clone, Debug)]
pub enum CurveType {
    /// 多项式.
    Polynomial {
        /// 多项式的次数.
        degree: u32,
    },

    /// 三次样条曲线.
    CubicSpline,
}

// Q: 用宏替代?

/// 拟合三次样条曲线.
///
/// `x` 是严格递增 (或递减) 的数组, `y` 是对应函数值, `k` 为两点之间的拟合点的数量.
pub fn cubic_spline_f32(x: ArrayView1<f32>, y: ArrayView1<f32>, k: u32) -> VecPair<f32> {
    cubic_spline::CubicSplineImp::<f32>::new(x.view(), y.view(), k).make_spline()
}

/// 拟合三次样条曲线.
///
/// `x` 是严格递增 (或递减) 的数组, `y` 是对应函数值, `k` 为两点之间的拟合点的数量.
pub fn cubic_spline_f64(x: ArrayView1<f64>, y: ArrayView1<f64>, k: u32) -> VecPair<f64> {
    cubic_spline::CubicSplineImp::<f64>::new(x.view(), y.view(), k).make_spline()
}

/// 基于最小二乘法拟合 n 次多项式曲线.
///
/// `x` 是自变量数组, `y` 是对应函数值, `degree` 是多项式次数 (最小为 1).
/// 曲线会拟合等距离的 `points` 个点 (`points` >= 3), 并保证区间端点能被选取.
pub fn polynomial_f64(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    degree: u32,
    points: u32,
) -> VecPair<f64> {
    polynomial::PolyImp::<f64>::new(x.view(), y.view(), degree, points).make_curve()
}

/// 基于最小二乘法拟合 n 次多项式曲线.
///
/// `x` 是自变量数组, `y` 是对应函数值, `degree` 是多项式次数 (最小为 1).
/// 曲线会拟合等距离的 `points` 个点 (`points` >= 3), 并保证区间端点能被选取.
pub fn polynomial_f32(
    x: ArrayView1<f32>,
    y: ArrayView1<f32>,
    degree: u32,
    points: u32,
) -> VecPair<f32> {
    polynomial::PolyImp::<f32>::new(x.view(), y.view(), degree, points).make_curve()
}
