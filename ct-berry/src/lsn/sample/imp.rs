//! 拟合取样的实际实现.
//!
//! 目前只支持自然 x-y 方向的多项式拟合 (样条曲线拟合不受影响).

use super::{CalcError, RawSurface, SampledCurve};
use crate::lsn::points::points_distance;
use crate::{Idx2d, Idx2dF};
use std::mem;

type R = Result<Vec<SampledCurve>, CalcError>;

/// `Vec<SampledCurve>` 生成器. 除此外不保存其它状态.
struct SampledCurvesAcc {
    acc: Vec<SampledCurve>,
    cur: SampledCurve,
}

impl SampledCurvesAcc {
    #[inline]
    pub fn new() -> Self {
        Self {
            acc: vec![],
            cur: SampledCurve::new(),
        }
    }

    /// 加入一个取样坐标.
    #[inline]
    pub fn add_sampled(&mut self, (x, y): Idx2dF) {
        self.cur.samp_x.push(x);
        self.cur.samp_y.push(y);
    }

    /// 设置/替换表面曲线坐标组.
    #[inline]
    pub fn add_surface(&mut self, (mut x, mut y): (Vec<f64>, Vec<f64>)) {
        debug_assert_eq!(x.len(), y.len());
        mem::swap(&mut self.cur.liver_x, &mut x);
        mem::swap(&mut self.cur.liver_y, &mut y);
    }

    /// 完成了一个取样.
    ///
    /// 正常情况下, 返回包括当前取样的所有取样个数.
    #[inline]
    pub fn finish_one(&mut self) -> Result<u8, CalcError> {
        let is_err = self.cur.samp_x.len() < 2;
        self.cur.shrink_to_fit();
        self.acc
            .push(mem::replace(&mut self.cur, SampledCurve::new()));
        if is_err {
            Err(CalcError::LengthTooShort)
        } else {
            Ok(self.acc.len() as u8)
        }
    }

    #[inline]
    pub fn num_finished(&self) -> u8 {
        self.acc.len() as u8
    }

    #[inline]
    pub fn consume(self) -> Vec<SampledCurve> {
        self.acc
    }
}

/// 等距离采样.
fn get_fixed(points: &[Idx2d], length_hint: f64, sampled: u32) -> Vec<usize> {
    // points.len() >= sampled >= 2.
    debug_assert!(points.len() as u32 >= sampled && sampled >= 2);

    let mut ans = Vec::with_capacity(sampled as usize);
    ans.push(0);

    let len_threshold = length_hint / (sampled - 1) as f64;
    let mut acc_len = 0.0;

    for (index, wnd) in points.windows(2).enumerate() {
        let [p1, p2] = wnd else { unreachable!() };
        let cur_len = points_distance(*p1, *p2);

        // 到达采样分界线
        if acc_len + cur_len >= len_threshold {
            ans.push(index + 1);
            acc_len = acc_len + cur_len - len_threshold;
            if ans.len() + 1 == sampled as usize {
                ans.push(points.len() - 1);
                return ans;
            }
        } else {
            acc_len += cur_len;
        }
    }
    unreachable!()
}

// 1
pub fn num_cont_eqd(surface: RawSurface, slt: f64, num: u8, dist: f64) -> R {
    // `num` 个无间隙连续 section, 每个 section 长度至少为 `slt`, 且每隔 `>=dist` 取一次样.

    // 曲线总长度.
    let length = surface.mm_length();

    // 曲线不够长就直接返回错误.
    if num as f64 * slt < length {
        return Err(CalcError::LengthTooShort);
    }

    // `len_threshold` > `slt`. 将边缘分割成 `num` 段,
    // 每段按照 "不超过 `len_threshold` 并尽可能长" 为标准切割 (超过也没问题).
    let len_threshold = length / (num as f64);

    debug_assert!(len_threshold > slt);

    // (累计取样距离, 累计 section 长度, 上一次下标)
    let (mut acc_dist, mut acc_len, mut last_idx) = (0.0, 0.0, 0usize);

    let mut acc = SampledCurvesAcc::new();
    let points = surface.points();

    // 逻辑比较复杂的单趟循环
    for (index, wnd) in points.windows(2).enumerate() {
        let [p1, p2] = wnd else { unreachable!() };
        let cur_len = points_distance(*p1, *p2);

        // 一般来说最后一个 section 总长度会偏大, 我们进行特殊处理.
        if acc.num_finished() + 1 == num {
            acc_len += cur_len;

            // 最后一对?
            if index + 1 == points.len() {
                assert!(acc_len >= len_threshold);

                // 判断是否到了下一个采样点
                if acc_dist + cur_len >= dist {
                    acc.add_sampled(surface.hwu2xy(*p2));
                }
                acc.add_surface(surface.as_xy(&points[last_idx..]));
                acc.finish_one()?;
                break; // 等价于 continue
            }

            // 判断是否到了下一个采样点
            if acc_dist + cur_len >= dist {
                acc.add_sampled(surface.hwu2xy(*p2));
                acc_dist = (acc_dist + cur_len) % dist;
            } else {
                acc_dist += cur_len;
            }
            continue;
        }

        // 判断是否到了下一个采样点
        let mut p2_sample_point = false;
        if acc_dist + cur_len >= dist {
            // acc.add_sampled(surface.hwu2xy(*p2)); 在后面才能判断加不加
            acc_dist = (acc_dist + cur_len) % dist;
            p2_sample_point = true;
        } else {
            acc_dist += cur_len;
        }

        // 判断是否到了下一个 section
        if acc_len + cur_len >= len_threshold {
            // 如果 `slt` 恰巧过小, 则可能导致边界情况. 要精确捕捉.
            if acc_len < slt {
                return Err(CalcError::LengthTooShort);
            }
            acc.add_surface(surface.as_xy(&points[last_idx..=index]));
            assert!(acc.finish_one()? < num);

            // 不强制采样一个新 section 的开端的第一个像素点
            acc_len = cur_len;
            acc_dist = cur_len % dist;
            last_idx = index; // 两个相邻 section 是连续的.
        } else {
            acc_len += cur_len;
            if p2_sample_point {
                acc.add_sampled(surface.hwu2xy(*p2));
            }
        }
    }
    assert_eq!(acc.num_finished(), num);
    // assert: acc.cur is empty.
    Ok(acc.consume())
}

// 2
pub fn num_cont_every(surface: RawSurface, slt: f64, num: u8, pn: u32) -> R {
    // `num` 个无间隙连续 section, 每个 section 长度至少为 `slt`, 且每隔 `pn`
    // 个表面像素取一次样. 模仿 (1) 即可.

    let length = surface.mm_length();
    if num as f64 * slt < length {
        return Err(CalcError::LengthTooShort);
    }
    let len_threshold = length / (num as f64);
    debug_assert!(len_threshold > slt);
    let (mut acc_n, mut acc_len, mut last_idx) = (0u32, 0.0, 0usize);

    let mut acc = SampledCurvesAcc::new();
    let points = surface.points();

    for (index, wnd) in points.windows(2).enumerate() {
        let [p1, p2] = wnd else { unreachable!() };
        let cur_len = points_distance(*p1, *p2);

        if acc.num_finished() + 1 == num {
            acc_len += cur_len;
            if index + 1 == points.len() {
                assert!(acc_len >= len_threshold);
                if acc_n + 1 == pn {
                    acc.add_sampled(surface.hwu2xy(*p2));
                }
                acc.add_surface(surface.as_xy(&points[last_idx..]));
                acc.finish_one()?;
                break;
            }
            if acc_n + 1 == pn {
                acc.add_sampled(surface.hwu2xy(*p2));
                acc_n = 0;
            } else {
                acc_n += 1;
            }
            continue;
        }

        let mut p2_sample_point = false;
        if acc_n + 1 == pn {
            acc_n = 0;
            p2_sample_point = true;
        } else {
            acc_n += 1;
        }

        if acc_len + cur_len >= len_threshold {
            if acc_len < slt {
                return Err(CalcError::LengthTooShort);
            }
            acc.add_surface(surface.as_xy(&points[last_idx..]));
            assert!(acc.finish_one()? < num);

            acc_len = cur_len;
            acc_n = 0;
            last_idx = index;
        } else {
            acc_len += cur_len;
            if p2_sample_point {
                acc.add_sampled(surface.hwu2xy(*p2));
            }
        }
    }
    assert_eq!(acc.num_finished(), num);
    Ok(acc.consume())
}

// 3
pub fn num_cont_fixed(surface: RawSurface, slt: f64, num: u8, pn: u32) -> R {
    // `num` 个无间隙连续 section, 每个 section 长度至少为 `slt`, 且进行等距采样.
    // 模仿 (1) 即可.

    let length = surface.mm_length();
    if num as f64 * slt < length {
        return Err(CalcError::LengthTooShort);
    }
    let len_threshold = length / (num as f64);
    debug_assert!(len_threshold > slt);
    let (mut acc_len, mut last_idx) = (0.0, 0usize);

    let mut acc = SampledCurvesAcc::new();
    let points = surface.points();

    for (index, wnd) in points.windows(2).enumerate() {
        if acc.num_finished() + 1 == num {
            for pos in get_fixed(&points[last_idx..], len_threshold, pn)
                .into_iter()
                .map(|i| surface.hwu2xy(points[last_idx + i]))
            {
                acc.add_sampled(pos);
            }
            acc.add_surface(surface.as_xy(&points[last_idx..]));
            acc.finish_one()?;
            break;
        }

        let [p1, p2] = wnd else { unreachable!() };
        let cur_len = points_distance(*p1, *p2);

        // 判断是否到了下一个 section
        if acc_len + cur_len >= len_threshold {
            // 如果 `slt` 恰巧过小, 则可能导致边界情况. 要精确捕捉.
            if acc_len < slt {
                return Err(CalcError::LengthTooShort);
            }
            for pos in get_fixed(&points[last_idx..], len_threshold, pn)
                .into_iter()
                .map(|i| surface.hwu2xy(points[last_idx + i]))
            {
                acc.add_sampled(pos);
            }
            acc.add_surface(surface.as_xy(&points[last_idx..=index]));
            assert!(acc.finish_one()? < num);

            acc_len = cur_len;
            last_idx = index;
        } else {
            acc_len += cur_len;
        }
    }
    assert_eq!(acc.num_finished(), num);
    Ok(acc.consume())
}

// 4
pub fn num_fixed_eqd(_surface: RawSurface, _slt: f64, _num: u8, _space: f64, _dist: f64) -> R {
    unimplemented!()
}

// 5
pub fn num_fixed_every(_surface: RawSurface, _slt: f64, _num: u8, _space: f64, _pn: u32) -> R {
    unimplemented!()
}

// 6
pub fn num_fixed_fixed(_surface: RawSurface, _slt: f64, _num: u8, _space: f64, _pn: u32) -> R {
    unimplemented!()
}

// 7
pub fn num_max_eqd(_surface: RawSurface, _slt: f64, _num: u8, _dist: f64) -> R {
    unimplemented!()
}

// 8
pub fn num_max_every(_surface: RawSurface, _slt: f64, _num: u8, _pn: u32) -> R {
    unimplemented!()
}

// 9
pub fn num_max_fixed(_surface: RawSurface, _slt: f64, _num: u8, _pn: u32) -> R {
    unimplemented!()
}

// 10
pub fn len_cont_eqd(_surface: RawSurface, _snt: u8, _length: f64, _dist: f64) -> R {
    unimplemented!()
}

// 11
pub fn len_cont_every(_surface: RawSurface, _snt: u8, _length: f64, _pn: u32) -> R {
    unimplemented!()
}

// 12
pub fn len_cont_fixed(_surface: RawSurface, _snt: u8, _length: f64, _pn: u32) -> R {
    unimplemented!()
}

// 13
pub fn len_fixed_eqd(_surface: RawSurface, _snt: u8, _length: f64, _space: f64, _dist: f64) -> R {
    unimplemented!()
}

// 14
pub fn len_fixed_every(_surface: RawSurface, _snt: u8, _length: f64, _space: f64, _pn: u32) -> R {
    unimplemented!()
}

// 15
pub fn len_fixed_fixed(_surface: RawSurface, _snt: u8, _length: f64, _space: f64, _pn: u32) -> R {
    unimplemented!()
}

// 16
pub fn len_max_eqd(_surface: RawSurface, _snt: u8, _length: f64, _dist: f64) -> R {
    unimplemented!()
}

// 17
pub fn len_max_every(_surface: RawSurface, _snt: u8, _length: f64, _pn: u32) -> R {
    unimplemented!()
}

// 18
pub fn len_max_fixed(_surface: RawSurface, _snt: u8, _length: f64, _pn: u32) -> R {
    unimplemented!()
}
