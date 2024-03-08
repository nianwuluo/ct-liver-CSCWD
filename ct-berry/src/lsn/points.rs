//! 图像坐标操作.

use crate::fitting::{cubic_spline_f64, polynomial_f64, CurveType};
use crate::lsn::{CalcError, CalcResult};
use crate::{Idx2d, Idx2dF};
use itertools::izip;
use ndarray::ArrayView1;
use std::mem;

/// 原始索引 + dim 表示的原始肝脏稀疏轮廓表面.
/// 同时保存原图像高以进行 `(h, w) -> (x, y)` 坐标变换.
pub struct RawSurface<'a> {
    /// 从头至尾以 8-邻接的轮廓 (注意一般不首尾相连).
    points: &'a [Idx2d],

    /// `img_height` 是为了做 (h, w) 与 (x, y) 坐标系互转换的.
    img_height: usize,

    /// 水平切片像素分辨率.
    dim: f64,
}

impl<'a> RawSurface<'a> {
    /// 初始化.
    pub fn new(points: &'a [Idx2d], img_height: usize, dim: f64) -> Self {
        assert!(points.len() >= 3);
        Self {
            points,
            img_height,
            dim,
        }
    }

    #[inline]
    pub fn points(&self) -> &'a [Idx2d] {
        self.points
    }

    /// 从一组 `(h, w)` 构建出 `([x], [y])`.
    pub fn as_xy(&self, section: &[Idx2d]) -> (Vec<f64>, Vec<f64>) {
        let mut vx = Vec::with_capacity(section.len());
        let mut vy = Vec::with_capacity(section.len());
        for p in section {
            let (px, py) = self.hwu2xy(*p);
            vx.push(px);
            vy.push(py);
        }
        (vx, vy)
    }

    /// `(h, w)` -> `(x, y)`.
    #[inline]
    pub fn hwu2xy(&self, (point_h, point_w): Idx2d) -> Idx2dF {
        self.hw2xy((point_h as f64, point_w as f64))
    }

    /// `(h, w)` -> `(x, y)`.
    #[inline]
    fn hw2xy(&self, (point_h, point_w): Idx2dF) -> Idx2dF {
        (point_w, self.img_height as f64 - point_h - 1.0)
    }

    /// 根据 `self.{points, dim}` 计算曲线长度 (单位: 毫米).
    pub fn mm_length(&self) -> f64 {
        self.dim
            * self
                .points
                .windows(2)
                .map(|w| points_distance(w[0], w[1]))
                .sum::<f64>()
    }
}

/// 计算两个点的欧几里得距离. 如果这两个点不以 8-邻接则 panic.
#[inline]
pub(crate) fn points_distance((a, b): Idx2d, (c, d): Idx2d) -> f64 {
    DistanceKind::new(a.abs_diff(c), b.abs_diff(d)).eval_euclid()
}

/// 曼哈顿距离到欧几里得距离的中转站.
#[derive(Debug, Copy, Clone)]
enum DistanceKind {
    One,
    SqrtTwo,
    NotEightConnected,
}

impl DistanceKind {
    /// 以曼哈顿距离初始化.
    #[inline]
    pub fn new(manhattan1: usize, manhattan2: usize) -> Self {
        match (manhattan1, manhattan2) {
            (1, 0) | (0, 1) => Self::One,
            (1, 1) => Self::SqrtTwo,
            _ => Self::NotEightConnected,
        }
    }

    /// 计算欧几里得距离. 如果两个点不以 8-相邻, 则 panic.
    #[inline]
    pub fn eval_euclid(&self) -> f64 {
        match self {
            DistanceKind::One => 1.0,
            DistanceKind::SqrtTwo => std::f64::consts::SQRT_2,
            DistanceKind::NotEightConnected => panic!("not 8-connected"),
        }
    }
}

/*
    拟合采样点 `samp_*` + 肝表面点 `liver_*` + 从肝表面采样出的曲线 `fit_*`.
    `samp_*` 是 `liver_*` 的子集.
    `samp_*` 仅仅是对 `liver_*` 中部分点的采样, 与具体拟合曲线选取无关.
*/
/// 拟合采样点 + 肝表面点 + 从肝表面采样出的曲线.
///
/// 索引模式为 `(x, y)`.
#[derive(Debug)]
pub struct SampledCurve {
    pub(crate) samp_x: Vec<f64>,
    pub(crate) samp_y: Vec<f64>,
    pub(crate) liver_x: Vec<f64>,
    pub(crate) liver_y: Vec<f64>,
    pub(crate) fit_x: Vec<f64>,
    pub(crate) fit_y: Vec<f64>,
}

impl SampledCurve {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            samp_x: Vec::with_capacity(2),
            samp_y: Vec::with_capacity(2),
            liver_x: vec![],
            liver_y: vec![],
            fit_x: vec![],
            fit_y: vec![],
        }
    }

    /// 样本点个数.
    #[inline]
    pub fn sample_len(&self) -> usize {
        self.samp_x.len()
    }

    /// 肝脏轮廓点个数.
    #[inline]
    pub fn liver_len(&self) -> usize {
        self.liver_x.len()
    }

    /// 拟合的点个数.
    #[inline]
    pub fn fit_len(&self) -> usize {
        self.fit_x.len()
    }

    /// 采样点 x 坐标.
    #[inline]
    pub fn sample_x(&self) -> &[f64] {
        self.samp_x.as_slice()
    }

    /// 采样点 y 坐标.
    #[inline]
    pub fn sample_y(&self) -> &[f64] {
        self.samp_y.as_slice()
    }

    /// 肝表面 x 坐标.
    #[inline]
    pub fn liver_x(&self) -> &[f64] {
        self.liver_x.as_slice()
    }

    /// 肝表面 y 坐标.
    #[inline]
    pub fn liver_y(&self) -> &[f64] {
        self.liver_y.as_slice()
    }

    /// 拟合点 x 坐标.
    #[inline]
    pub fn fit_x(&self) -> &[f64] {
        self.fit_x.as_slice()
    }

    /// 拟合点 y 坐标.
    #[inline]
    pub fn fit_y(&self) -> &[f64] {
        self.fit_y.as_slice()
    }

    /// 清理多余占用的空间.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.samp_x.shrink_to_fit();
        self.samp_y.shrink_to_fit();
        self.liver_x.shrink_to_fit();
        self.liver_y.shrink_to_fit();
    }

    /// 假设像素分辨率单位为 1.0, 计算肝表面曲线的曲线纯数学长度.
    fn raw_liver_length(&self) -> f64 {
        // (x1, y1, x2, y2) iterator
        izip!(
            &self.liver_x,
            &self.liver_y,
            self.liver_x.iter().skip(1),
            self.liver_y.iter().skip(1)
        )
        .map(|(&x1, &y1, &x2, &y2)| ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt())
        .sum()
    }

    /// `(x, y)` -> `(h, w)`.
    #[inline]
    fn xy2hw_point((point_x, point_y): Idx2dF, img_height: usize) -> Idx2dF {
        (-point_y + img_height as f64 - 1.0, point_x)
    }

    /// `(x, y)` -> `(h, w)`.
    #[inline]
    fn xy2hw_point_inplace(x: &mut f64, y: &mut f64, img_height: usize) {
        (*x, *y) = Self::xy2hw_point((*x, *y), img_height);
    }

    /// `([x], [y])` -> `([h], [w])`.
    #[inline]
    fn xy2hw_vec_inplace(vx: &mut [f64], vy: &mut [f64], img_height: usize) {
        for (x, y) in izip!(vx.iter_mut(), vy.iter_mut()) {
            Self::xy2hw_point_inplace(x, y, img_height);
        }
    }

    /// 将内部表示从 `(x, y)` 格式转变为 `(h, w)` 格式. 原图像高度为 `img_height`.
    ///
    /// 如果 `sample_liver` 为 `true`, 则转换 sample 和 liver 的结果;
    /// 如果 `fit` 为 `true`, 则转换 fit 的结果. 两者可自由组合.
    ///
    /// # 注意
    ///
    /// 如果被转换的数据已经是 `(h, w)` 的格式, 则程序行为未定义.
    pub fn xy2hw_inplace(&mut self, sample_and_liver: bool, fit: bool, img_height: usize) {
        if sample_and_liver {
            Self::xy2hw_vec_inplace(&mut self.samp_x, &mut self.samp_y, img_height);
            Self::xy2hw_vec_inplace(&mut self.liver_x, &mut self.liver_y, img_height);
        }
        if fit {
            Self::xy2hw_vec_inplace(&mut self.fit_x, &mut self.fit_y, img_height);
        }
    }

    /// 计算并填充/替换 `self.fit_*`.
    ///
    /// 该函数应当是在上游确定了 sample 规格后调用的.
    pub(crate) fn fit(
        &mut self,
        target_curve: CurveType,
        sample_per_mm: u32,
        dim: f64,
    ) -> CalcResult<()> {
        let samp_len = self.sample_len();
        debug_assert!(samp_len >= 3);

        // 总共采样点数
        let points = (self.raw_liver_length() * dim * (sample_per_mm as f64)) as u32;

        let x_view = ArrayView1::from_shape([samp_len], self.sample_x()).unwrap();
        let y_view = ArrayView1::from_shape([samp_len], self.sample_y()).unwrap();

        let (mut fit_x, mut fit_y) = match target_curve {
            CurveType::Polynomial { degree } => {
                if degree >= samp_len as u32 {
                    return Err(CalcError::TooFewSamples(samp_len as u32, degree + 1));
                }
                polynomial_f64(x_view, y_view, degree, points)
            }
            CurveType::CubicSpline => cubic_spline_f64(x_view, y_view, points),
        };

        mem::swap(&mut self.fit_x, &mut fit_x);
        mem::swap(&mut self.fit_y, &mut fit_y);
        Ok(())
    }

    /// 计算肝表面结节 (Liver Surface Nodularity) 评分.
    ///
    /// LSN 评分被定义为肝脏表面每个点到拟合曲线的距离的平均值.
    pub fn lsn(&self) -> f64 {
        use ordered_float::NotNan;

        let mut lsn = 0.0;
        for (lx, ly) in Self::point_f64_iter(&self.liver_x, &self.liver_y) {
            let (mx, my) = Self::point_f64_iter(&self.fit_x, &self.fit_y)
                .min_by_key(|&(fx, fy)| {
                    NotNan::<f64>::new((fx - lx).powi(2) + (fy - ly).powi(2)).unwrap()
                })
                .unwrap();
            lsn += ((mx - lx).powi(2) + (my - ly).powi(2)).sqrt();
        }
        lsn / self.liver_len() as f64
    }

    #[inline]
    fn point_f64_iter<'a>(x: &'a [f64], y: &'a [f64]) -> impl Iterator<Item = Idx2dF> + 'a {
        izip!(x.iter().copied(), y.iter().copied())
    }
}
