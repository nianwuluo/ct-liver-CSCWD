//! CT 水平切片上的扇区.
//!
//! 我们一般使用行优先编码存储二维图像. 其中行就是 "Height" (垂直方向), 列就是 "Width" (水平方向).
//! 然后将 "Height" 作为平面直角坐标系中的 x 轴, 将 "Width" 作为平面直角坐标系中的 y 轴,
//! 这样相当于将原先的平面直角坐标系按顺时针旋转了 90 度.
//!
//! 以 `(0, 0)` 为原点, 则平面上任意点对的方向就可以通过 `atan2` 确定下来了.
//! 注意我们通过归一化保证了弧度的范围为 `[0, 2 * PI)`.

use nifti::NiftiHeader;
use num::ToPrimitive;
use std::fmt::Formatter;

use crate::Idx2d;

type Idx2dI32 = (i32, i32);

const PI_2: f64 = std::f64::consts::PI * 2.0;

/// 二维图像上的一个扇区, 由顶点和两条射线 (通过弧度表示) 组成.
///
/// 该结构不负责检测图像越界.
#[derive(Copy, Clone)]
pub struct Sector {
    /// 中心坐标
    center: Idx2dI32,
    /// [0, 2 * pi)
    arc1: f64,
    /// [0, 2 * pi); `arc1 -> 逆时针 -> arc2`
    arc2: f64,
}

/// 弧度转换为角度.
fn arc_to_angle(arc: f64) -> f64 {
    arc * 180.0 * std::f64::consts::FRAC_1_PI
}

/// 角度转换为弧度.
#[inline]
fn angle_to_arc(angle: f64) -> f64 {
    angle * std::f64::consts::PI / 180.0
}

/// 内部会将弧度转换为角度, 因为角度更加直观. 另外压缩到一行.
impl std::fmt::Debug for Sector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Sector {{ center: {:?}, arc1: {:.4}°, arc2: {:.4}° }}",
            self.center,
            arc_to_angle(self.arc1),
            arc_to_angle(self.arc2)
        ))
    }
}

/// `Sector` 初始化错误.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InitSectorError {
    /// 原点无法用 `f64` 精确表示.
    CenterOutOfBound,
    /// 弧度超出表示范围.
    ArcOutOfRange,
    /// 空范围. 该情况不在 `Sector` 的考虑范围内.
    EmptyRange,
}

impl Sector {
    /// 以 `center` 为原点, `arc1` 和 `arc2`
    /// 分别为两条射线相对于原点的弧度, 创建一个扇区.
    /// 这个扇区扫过的区域被认定为从 `arc1` 出发,
    /// 通过 **逆时针** 方向前进直到触碰到 `arc2` 所经过的所有区域.
    ///
    /// # 返回值
    ///
    /// - 当 `center` 无法用 `i32` 精确表示时, 返回 `Err(SectorError::CenterOutOfBound)`;
    /// - 当 `arc1` 或 `arc2` 不在 `[0, 2 * PI)` 范围内时, 返回 `Err(SectorError::ArcOutOfRange)`;
    /// - 当 `arc1 == arc2` 时, 返回 `Err(Sector::EmptyRange)`;
    /// - 其他情况下成功, 返回 `Ok(Sector)`.
    pub fn new(center: Idx2d, arc1: f64, arc2: f64) -> Result<Self, InitSectorError> {
        let center = Self::usize_to_i32_2d(&center).ok_or(InitSectorError::CenterOutOfBound)?;
        const R: std::ops::Range<f64> = 0.0..PI_2;
        if !R.contains(&arc1) || !R.contains(&arc2) {
            return Err(InitSectorError::ArcOutOfRange);
        }
        if arc1 == arc2 {
            return Err(InitSectorError::EmptyRange);
        }
        Ok(Self { center, arc1, arc2 })
    }

    /// 创建全区域扇区 (圆).
    ///
    /// 注意该方法是必要的, 因为 `Self::new` 只能创建一个严格的扇区 (2 * pi) 值不被允许.
    pub fn new_circle(center: Idx2d) -> Result<Self, InitSectorError> {
        let center = Self::usize_to_i32_2d(&center).ok_or(InitSectorError::CenterOutOfBound)?;
        Ok(Self {
            center,
            arc1: 0.0,
            arc2: PI_2,
        })
    }

    /// 获取中心点.
    #[inline]
    pub fn center(&self) -> Idx2dI32 {
        self.center
    }

    /// 获取中心点的第一个分量.
    #[inline]
    pub fn height(&self) -> i32 {
        self.center.0
    }

    /// 获取中心点的第二个分量.
    #[inline]
    pub fn width(&self) -> i32 {
        self.center.1
    }

    /// 判断点 `point` 是否被包含在扇区中.
    pub fn contains(&self, point: Idx2d) -> bool {
        let Some(p) = Self::usize_to_i32_2d(&point) else {
            return false;
        };
        self.clamp(self.arc_to(p))
    }

    /// 获取本扇区的弧度.
    pub fn arc(&self) -> f64 {
        match self.arc2 - self.arc1 {
            d if d > 0.0 => d,
            d => {
                debug_assert_ne!(d, 0.0);
                PI_2 + d
            }
        }
    }

    /// 获取本扇区的角度.
    #[inline]
    pub fn angle(&self) -> f64 {
        arc_to_angle(self.arc())
    }

    /// 试将 `(usize, usize)` 转换为 `(i32, i32)`.
    /// 如果越界则返回 `None`.
    #[inline]
    fn usize_to_i32_2d((p1, p2): &Idx2d) -> Option<Idx2dI32> {
        let p1 = (*p1).to_i32()?;
        let p2 = (*p2).to_i32()?;
        Some((p1, p2))
    }

    /// 获取点 `(h, w)` 相对于 `self.center` 的弧度.
    /// 该弧度取值范围为 `[0, 2 * PI)`.
    ///
    /// # 弧度规范
    ///
    /// - h 增加的方向弧度为 `0`;
    /// - w 增加的方向弧度为 `pi / 2`;
    /// - h 减少的方向弧度为 `pi`;
    /// - w 减少的方向弧度为 `3 * pi / 2`;
    fn arc_to(&self, (h, w): Idx2dI32) -> f64 {
        let h = (h - self.height()) as f64;
        let w = (w - self.width()) as f64;
        let mut raw = f64::atan2(w, h);
        if raw < 0.0 {
            raw += PI_2;
        }
        raw
    }

    /// 判断是否存在逆时针关系 `self.arc1` ->(le) `arc` ->(le) `self.arc2`.
    #[inline]
    fn clamp(&self, arc: f64) -> bool {
        if self.is_circle() {
            return true;
        } else if !arc.is_finite() || !(0.0..PI_2).contains(&arc) {
            return false;
        }

        if self.arc1 < self.arc2 {
            (self.arc1..=self.arc2).contains(&arc)
        } else {
            debug_assert!(self.arc2 < self.arc1);
            arc == self.arc2 || !(self.arc2..self.arc1).contains(&arc)
        }
    }

    /// 该扇区是否是一个圆 (特殊情况)?
    #[inline]
    fn is_circle(&self) -> bool {
        self.arc2 == PI_2
    }
}

// 下面的所有项: 描述在一组连续的 CT scans 上 LLS 的定位扇区模式.

/// [`Sector`] 中相对固定一侧的射线是什么方向?
///
/// # 注意
///
/// 方向是由 "Height-O-Width", 而不是由自然 "x-O-y" 坐标系决定的.
/// 具体来说, 以二维图像 `(10, 10)` 为原点:
///
/// 1. `(11, 10)` 为 `HeightPos` 方向;
/// 2. `(9, 10)` 为 `HeightNeg` 方向;
/// 3. `(10, 11)` 为 `WidthPos` 方向;
/// 4. `(10, 9)` 为 `WidthNeg` 方向.
#[derive(Copy, Clone, Debug)]
pub(crate) enum AxisDirection {
    /// "Height" 增加的方向 (x 轴正方向).
    HeightPos,

    /// "Height" 减少的方向 (x 轴负方向).
    HeightNeg,

    /// "Width" 增加的方向 (y 轴正方向).
    #[allow(dead_code)]
    WidthPos,

    /// "Width" 减少的方向 (y 轴负方向).
    #[allow(dead_code)]
    WidthNeg,
}

/// 在一个 [`Sector`] 上 [`AxisDirection`] 所在的射线出发,
/// 另一端射线处于顺时针还是逆时针方向?
#[derive(Copy, Clone, Debug)]
pub(crate) enum Orientation {
    /// 顺时针方向.
    Clockwise,

    /// 逆时针方向.
    CounterClockwise,
}

// impl Orientation {
//     /// 是否是顺时针方向?
//     #[inline]
//     pub fn is_clockwise(&self) -> bool {
//         matches!(self, Self::Clockwise)
//     }
//
//     /// 是否是逆时针方向?
//     #[inline]
//     pub fn is_counter_clockwise(&self) -> bool {
//         !self.is_clockwise()
//     }
// }

/// 描述专用于 LLS 的 [`Sector`] 元信息.
#[derive(Copy, Clone)]
pub struct LlsSectorPattern {
    /// 固定射线方向.
    axis: AxisDirection,

    /// 另一条射线 **相对于固定射线 `axis`** 的时针方向.
    o: Orientation,
}

impl std::fmt::Debug for LlsSectorPattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "LlsSectorPattern {{ AxisDirection::{:?}, Orientation::{:?} }}",
            self.axis, self.o
        ))
    }
}

impl LlsSectorPattern {
    /// 从 nifti 元数据 `header` 中构建 LLS [`Sector`] 元信息.
    ///
    /// # 返回值
    ///
    /// 1. 如果 `header` 的 `quatern*` 属性不是轴向单位向量, 则返回
    ///   `Err(InitLlsPatternError::NotAxisVector)`.
    /// 2. 如果 `header` 的 `quatern*` 所描述的方向目前未知, 则返回
    ///   `Err(InitLlsPatternError::Unknown(*))`, 其中包含了未知数据组.
    pub fn from_header(header: &NiftiHeader) -> Result<Self, InitLlsPatternError> {
        fn f2i8(f: f32) -> Option<i8> {
            let r = f.round();
            ((f - r).abs() < 1e-9).then_some(r as i8)
        }

        macro_rules! init_pattern {
            (HeightPos, CounterClockwise) => {
                LlsSectorPattern {
                    axis: AxisDirection::HeightPos,
                    o: Orientation::CounterClockwise,
                }
            };
            (HeightPos, Clockwise) => {
                LlsSectorPattern {
                    axis: AxisDirection::HeightPos,
                    o: Orientation::Clockwise,
                }
            };
            (HeightNeg, Clockwise) => {
                LlsSectorPattern {
                    axis: AxisDirection::HeightNeg,
                    o: Orientation::Clockwise,
                }
            };
        }

        let qb = f2i8(header.quatern_b).ok_or(InitLlsPatternError::NotAxisVector)?;
        let qc = f2i8(header.quatern_c).ok_or(InitLlsPatternError::NotAxisVector)?;
        let qd = f2i8(header.quatern_d).ok_or(InitLlsPatternError::NotAxisVector)?;
        let qform = header.qform_code;

        match (qform, (qb, qc, qd)) {
            (0, (0, 0, 0)) | (1, (0, 1, 0)) | (2, (0, 1, 0)) => {
                Ok(init_pattern!(HeightPos, CounterClockwise))
            }
            (2, (0, 0, 0)) => Ok(init_pattern!(HeightPos, Clockwise)),
            (2, (0, 0, 1)) => Ok(init_pattern!(HeightNeg, Clockwise)),
            (qform, tup) => Err(InitLlsPatternError::Unknown(qform, tup)),
        }
    }

    /// 基于弧度构建扇区.
    ///
    /// `center` 是扇区中心. 参数 `offset_arc` 是一个弧度,
    /// 所量化的是期望扇区所真包围的唯一水平/垂直方向到非固定自由射线的弧度数量.
    /// 该角度的值根据 LLS 的定位经验, 应该位于 `(0.0, 2.0 * PI / 3.0]` 之间,
    /// 否则函数 panic.
    pub fn build_from_arc(&self, center: Idx2d, offset_arc: f64) -> Sector {
        use std::f64::consts::*;
        const ANGLE_120: f64 = 2.0 * FRAC_PI_3; // 120 度

        assert!(
            0.0 < offset_arc && offset_arc <= ANGLE_120,
            "弧度 `{offset_arc}` 越界"
        );

        // 注意 arc1 -> 逆时针 -> arc2.
        let (arc1, arc2) = match (self.axis, self.o) {
            // 0..=52 in LiTS
            (AxisDirection::HeightPos, Orientation::CounterClockwise) => (0.0, offset_arc),

            // 68..=82 in LiTS
            (AxisDirection::HeightPos, Orientation::Clockwise) => (PI_2 - offset_arc, 0.0),

            // 53..=67, 83..=130
            (AxisDirection::HeightNeg, Orientation::Clockwise) => (PI - offset_arc, PI),
            _ => unreachable!(),
        };

        let ans = Sector::new(center, arc1, arc2).unwrap();
        debug_assert!(ans.angle() < 180.0);
        ans
    }

    /// 基于夹角构建扇区.
    ///
    /// `center` 是扇区中心. 参数 `offset_angle` 是一个夹角,
    /// 所量化的是期望扇区所真包围的唯一水平/垂直方向到非固定自由射线的角度数量.
    /// 该角度的值根据 LLS 的定位经验, 应该位于 `(0.0, 120.0]`° 之间,
    /// 否则函数 panic.
    #[inline]
    pub fn build_from_angle(&self, center: Idx2d, offset_angle: f64) -> Sector {
        self.build_from_arc(center, angle_to_arc(offset_angle))
    }

    /// 返回固定射线方向.
    #[inline]
    pub(crate) fn quadrant(&self) -> AxisDirection {
        self.axis
    }

    /// 返回从固定射线起始的时针方向.
    #[inline]
    pub(crate) fn orientation(&self) -> Orientation {
        self.o
    }
}

/// 初始化 [`LlsSectorPattern`] 错误.
#[derive(Clone, Debug)]
pub enum InitLlsPatternError {
    /// 非轴向单位向量.
    NotAxisVector,

    /// 未知方向. `(qform_code, quatern_b, quatern_c, quatern_c)`
    Unknown(i16, (i8, i8, i8)),
}

#[cfg(test)]
mod tests {
    use super::{Idx2dI32, InitSectorError, Sector, PI_2};
    use crate::Idx2d;
    use std::f64::consts::*;

    fn f64_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-8
    }

    /// 测试基本初始化错误问题.
    #[test]
    fn test_sector_init_err() {
        let s = Sector::new((0, usize::MAX), 0.0, 1.0).unwrap_err();
        assert_eq!(s, InitSectorError::CenterOutOfBound);

        let s = Sector::new((1, 1), 1.0, PI_2).unwrap_err();
        assert_eq!(s, InitSectorError::ArcOutOfRange);
        let s = Sector::new((1, 1), -0.01, 1.0).unwrap_err();
        assert_eq!(s, InitSectorError::ArcOutOfRange);

        let s = Sector::new((3, 4), 1.2, 1.2).unwrap_err();
        assert_eq!(s, InitSectorError::EmptyRange);
    }

    fn assert_arc(from: Idx2d, to: Idx2dI32, arc: f64) {
        let s = Sector::new_circle(from).unwrap();
        assert!(f64_eq(arc, s.arc_to(to)));
    }

    /// 测试基本角度的正确性.
    #[test]
    fn test_sector_arc() {
        // corner case
        assert_arc((0, 0), (0, 0), 0.0);

        assert_arc((1, 1), (2, 1), 0.0);
        assert_arc((1, 1), (2, 2), FRAC_PI_4);
        assert_arc((1, 1), (1, 2), FRAC_PI_2);
        assert_arc((1, 1), (0, 2), FRAC_PI_2 + FRAC_PI_4);
        assert_arc((1, 1), (0, 1), PI);
        assert_arc((1, 1), (0, 0), PI + FRAC_PI_4);
        assert_arc((1, 1), (1, 0), PI + FRAC_PI_2);
        assert_arc((1, 1), (2, 0), PI_2 - FRAC_PI_4);
    }

    /// 创建一个 30 度到 60 度的扇区并进行基本测试.
    #[test]
    fn test_sector_no_across() {
        let (ch, cw) = (10, 10);
        let s = Sector::new((ch, cw), FRAC_PI_6 - 1e-8, FRAC_PI_3 + 1e-8).unwrap();
        let sqrt_3 = 3.0f64.sqrt();
        let (lb, ub) = (sqrt_3 / 3.0, sqrt_3);

        for (h, w) in (0usize..=20)
            .flat_map(move |first| (0usize..=20).map(move |second| (first, second)))
            .filter(|&(h, w)| h != 10 && w != 10)
        {
            let tan_val = (w as f64 - cw as f64) / (h as f64 - ch as f64);
            println!("({h}, {w}) -> tan: {tan_val}");

            match (lb..=ub).contains(&tan_val) {
                true if h > 10 => assert!(s.contains((h, w))),
                _ => assert!(!s.contains((h, w))),
            }
        }
    }

    /// 创建一个 -30 度到 30 度的扇区并进行基本测试.
    #[test]
    fn test_sector_across() {
        let (ch, cw) = (10, 10);
        let s = Sector::new((ch, cw), PI_2 - FRAC_PI_6 - 1e-8, FRAC_PI_6 + 1e-8).unwrap();
        let sqrt_3 = 3.0f64.sqrt();
        let (lb, ub) = (-sqrt_3 / 3.0, sqrt_3 / 3.0);

        for (h, w) in (0usize..=20)
            .flat_map(move |first| (0usize..=20).map(move |second| (first, second)))
            .filter(|&(h, w)| h != 10 && w != 10)
        {
            let tan_val = (w as f64 - cw as f64) / (h as f64 - ch as f64);
            println!("({h}, {w}) -> tan: {tan_val}");

            match (lb..=ub).contains(&tan_val) {
                true if h > 10 => assert!(s.contains((h, w))),
                _ => assert!(!s.contains((h, w))),
            }
        }
    }
}
