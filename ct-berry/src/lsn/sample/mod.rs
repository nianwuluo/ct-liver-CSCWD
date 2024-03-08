//! 从肝表面的左肝外区 (LLS) 提取拟合曲线.

mod imp;

use super::{CalcError, CalcResult, RawSurface, SampledCurve};
use crate::fitting::CurveType;
use crate::Idx2d;
use either::Either;

/// 描述 **section 之间** 关系的参数.
#[derive(Debug, Clone, Copy)]
pub enum Spacing {
    // 当基于 `NumBased` 时, 所有 section 将占满肝脏表面.
    // 当基于 `LenBased` 时, 所有 section 将 "挤" 在肝脏表面的中心.
    /// 几何上连续.
    Contiguous,

    /// 固定距离 (单位: 毫米). 当肝脏表面长度超出时, 将舍弃一定量的的两头部分.
    Fixed(f64),

    /// 尽量保持最大距离.
    Maximum,
}

/// 每一个 section 的曲线采样规则.
#[derive(Debug, Clone, Copy)]
pub enum SampleRule {
    /// 等距采样.
    EqualDistance(f64),

    /// 每隔给定个数的点进行采样.
    EveryPoints(u32),

    /// 采样固定个数的点.
    FixedPoints(u32),
}

/// 基于 section 数量的规则.
#[derive(Debug, Clone, Copy)]
struct NumBased {
    /// (单位: 毫米) 不接受 section 长度小于本字段的的解.
    section_length_threshold: f64,

    /// 所需要的 section 的 **精确** 数量.
    num: u8,
}

/// 基于 section 长度的规则.
#[derive(Debug, Clone, Copy)]
struct LenBased {
    /// 不接受 section 个数小于本字段的解.
    section_num_threshold: u8,

    /// (单位: 毫米) 每一个 section 的 **精确** 长度.
    length: f64,
}

/// 采样规则. 实际计算 LSN 时所需的输入参数.
#[derive(Debug, Clone)]
pub struct SampleSpec {
    /// 两个相邻 section 之间的关系.
    spacing: Spacing,

    // 注: 这里直接用 `Either` 是考虑今后扩展的可能性不大.
    /// 采样大规则.
    manner: Either<NumBased, LenBased>,

    /// 采样小规则.
    rule: SampleRule,

    /// 像素在水平和垂直方向的分辨率 (单位: 毫米).
    dim: f64,

    /// 希望拟合的曲线类型.
    curve_type: CurveType,

    /// 拟合曲线中, 每毫米的采样点个数.
    sample_per_mm: u32,
}

impl SampleSpec {
    /// 基于固定 section 数量的规则构建参数.
    ///
    /// 指定 section 数量精确值为 `section_num`.
    /// 输入要求按照 `spacing` 间隔规则, `rule` 采样规则,
    /// 且每 section 长度至少为 `section_length_threshold`,
    /// 像素在水平和垂直方向的分辨率为 `dim` (单位: 毫米),
    /// 要拟合的曲线类型为 `curve_type`,
    /// 拟合曲线中每毫米的采样点个数为 `sample_per_mm`.
    ///
    /// 如果存在非法参数, 则程序 panic.
    pub fn with_fixed_num(
        spacing: Spacing,
        section_num: u8,
        section_length_threshold: f64,
        rule: SampleRule,
        dim: f64,
        curve_type: CurveType,
        sample_per_mm: u32,
    ) -> Self {
        Self::assert_args(
            section_length_threshold,
            section_num,
            rule,
            dim,
            sample_per_mm,
        );

        Self {
            spacing,
            manner: Either::Left(NumBased {
                section_length_threshold,
                num: section_num,
            }),
            rule,
            dim,
            curve_type,
            sample_per_mm,
        }
    }

    /// 基于固定 section 长度的规则构建参数.
    ///
    /// 指定 section 长度精确值为 `section_length` (单位: 毫米).
    /// 输入要求按照 `spacing` 间隔规则, `rule` 采样规则,
    /// 至少有 `section_num_threshold` 个 section,
    /// 像素在水平和垂直方向的分辨率为 `dim` (单位: 毫米),
    /// 要拟合的曲线类型为 `curve_type`,
    /// 拟合曲线中每毫米的采样点个数为 `sample_per_mm`.
    ///
    /// 如果存在非法参数, 则程序 panic.
    pub fn with_fixed_length(
        spacing: Spacing,
        section_length: f64,
        section_num_threshold: u8,
        rule: SampleRule,
        dim: f64,
        curve_type: CurveType,
        sample_per_mm: u32,
    ) -> Self {
        Self::assert_args(
            section_length,
            section_num_threshold,
            rule,
            dim,
            sample_per_mm,
        );

        Self {
            spacing,
            manner: Either::Right(LenBased {
                section_num_threshold,
                length: section_length,
            }),
            rule,
            dim,
            curve_type,
            sample_per_mm,
        }
    }

    #[inline]
    fn assert_args(section_length: f64, section_num: u8, rule: SampleRule, dim: f64, spm: u32) {
        assert!(section_length > 0.0);
        assert_ne!(section_num, 0); // 根据医学研究, 一般不应小于 3
        match rule {
            SampleRule::EqualDistance(f) => {
                assert!(f < section_length);
            }
            SampleRule::EveryPoints(num) => {
                assert!(num > 1);
            }
            SampleRule::FixedPoints(num) => {
                assert!(num > 1);
            }
        }
        assert!(dim > 0.0);
        assert!((1..=10000).contains(&spm));
    }

    /// 获得肝脏曲线、肝脏曲线采样、拟合曲线采样.
    ///
    /// `points` 是肝左外区的 8-邻域稀疏曲线, `img_height` 是图像的高,
    pub fn sample(&self, points: &[Idx2d], img_height: usize) -> CalcResult<Vec<SampledCurve>> {
        // 目前 `self` 的参数都是合法的; `points` 不确定, 可以在运行时继续检查.
        assert!(points.len() >= 3); // 实际上应该远大于 3.

        let mut curves = self.sample_raw(points, img_height)?;

        // 以上法获得的曲线都还没有填充 `fit_*` 字段.
        for c in curves.iter_mut() {
            c.fit(self.curve_type, self.sample_per_mm, self.dim)?;
        }
        Ok(curves)
    }

    /// 获得肝脏曲线、肝脏曲线采样.
    ///
    /// `points` 是肝左外区的 8-邻域稀疏曲线, `img_height` 是图像的高.
    fn sample_raw(&self, points: &[Idx2d], img_height: usize) -> CalcResult<Vec<SampledCurve>> {
        let surface = RawSurface::new(points, img_height, self.dim);

        match (self.manner, self.spacing, self.rule) {
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Contiguous,
                SampleRule::EqualDistance(dist),
            ) => {
                imp::num_cont_eqd(surface, slt, num, dist) // 1
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Contiguous,
                SampleRule::EveryPoints(pn),
            ) => {
                imp::num_cont_every(surface, slt, num, pn) // 2
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Contiguous,
                SampleRule::FixedPoints(pn),
            ) => {
                imp::num_cont_fixed(surface, slt, num, pn) // 3
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Fixed(space),
                SampleRule::EqualDistance(dist),
            ) => {
                imp::num_fixed_eqd(surface, slt, num, space, dist) // 4
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Fixed(space),
                SampleRule::EveryPoints(pn),
            ) => {
                imp::num_fixed_every(surface, slt, num, space, pn) // 5
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Fixed(space),
                SampleRule::FixedPoints(pn),
            ) => {
                imp::num_fixed_fixed(surface, slt, num, space, pn) // 6
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Maximum,
                SampleRule::EqualDistance(dist),
            ) => {
                imp::num_max_eqd(surface, slt, num, dist) // 7
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Maximum,
                SampleRule::EveryPoints(pn),
            ) => {
                imp::num_max_every(surface, slt, num, pn) // 8
            }
            (
                Either::Left(NumBased {
                    section_length_threshold: slt,
                    num,
                }),
                Spacing::Maximum,
                SampleRule::FixedPoints(pn),
            ) => {
                imp::num_max_fixed(surface, slt, num, pn) // 9
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Contiguous,
                SampleRule::EqualDistance(dist),
            ) => {
                imp::len_cont_eqd(surface, snt, length, dist) // 10
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Contiguous,
                SampleRule::EveryPoints(pn),
            ) => {
                imp::len_cont_every(surface, snt, length, pn) // 11
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Contiguous,
                SampleRule::FixedPoints(pn),
            ) => {
                imp::len_cont_fixed(surface, snt, length, pn) // 12
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Fixed(space),
                SampleRule::EqualDistance(dist),
            ) => {
                imp::len_fixed_eqd(surface, snt, length, space, dist) // 13
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Fixed(space),
                SampleRule::EveryPoints(pn),
            ) => {
                imp::len_fixed_every(surface, snt, length, space, pn) // 14
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Fixed(space),
                SampleRule::FixedPoints(pn),
            ) => {
                imp::len_fixed_fixed(surface, snt, length, space, pn) // 15
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Maximum,
                SampleRule::EqualDistance(dist),
            ) => {
                imp::len_max_eqd(surface, snt, length, dist) // 16
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Maximum,
                SampleRule::EveryPoints(pn),
            ) => {
                imp::len_max_every(surface, snt, length, pn) // 17
            }
            (
                Either::Right(LenBased {
                    section_num_threshold: snt,
                    length,
                }),
                Spacing::Maximum,
                SampleRule::FixedPoints(pn),
            ) => {
                imp::len_max_fixed(surface, snt, length, pn) // 18
            }
        }
    }
}
