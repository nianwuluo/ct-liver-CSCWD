//! 2D 肝脏 CT 水平切片的肝左外区定位 (后处理).
//!
//! 该模块实现的功能是: 根据有序 4-邻接轮廓索引组和中心点,
//! 提取可能是肝左外区 (Left Lateral Segment) 的区域.

use crate::sector::Sector;
use crate::Idx2d;

/// 判断首尾相连的简单多边形是否是以顺时针连接的.
///
/// # 注意
///
/// 1. 多边形必须至少有 3 个顶点.
/// 2. 顺时针的概念同时适用于图像坐标系和自然坐标系, 无二义性.
/// 3. 图形必须是简单多边形, 不能有相交边, 也不能在同一条直线上,
///   否则结果无意义.
/// 4. 函数可能会计算溢出,
///   因此请考虑让数据规模和坐标的绝对值大小处于合理范围内.
pub fn is_clockwise_polygon(circle: &[Idx2d]) -> bool {
    assert!(circle.len() > 2);

    #[inline]
    fn shoelace(wnd: &[Idx2d]) -> f64 {
        let &[(x1, y1), (x2, y2)] = wnd else {
            unreachable!()
        };
        let (x1, y1, x2, y2) = (x1 as i64, y1 as i64, x2 as i64, y2 as i64);
        (x1 * y2 - x2 * y1) as f64
    }

    // 鞋带公式
    circle.windows(2).map(shoelace).sum::<f64>() < 0.0
}

/// 从 8-相邻且首尾相连的 `circle_surface` 索引组中提取出位于
/// `sector` 扇区的索引组.
///
/// # 注意
///
/// - 轮廓可能会 "折返", 因此在 `sector` 内的所有索引不一定能构成一个邻接的轮廓.
///   该函数会在邻接的轮廓中提取出最长的一个.
/// - `circle_surface` 不能存在越界索引, 否则程序会 panic.
/// - `circle_surface` 必须是从头至尾 8-邻接的, 否则程序行为未定义.
pub fn locate_lls(circle_surface: &[Idx2d], sector: Sector) -> Vec<Idx2d> {
    if circle_surface.is_empty() {
        return vec![];
    }
    let first = circle_surface[0];
    if circle_surface.len() == 1 {
        return if sector.contains(first) {
            vec![first]
        } else {
            vec![]
        };
    }

    // 最关键的是要处理好环的问题 (由 `RingState` 来解决).
    let mut state = RingState::new();

    for (index, pos) in circle_surface
        .iter()
        .chain(circle_surface.iter().take_while(|&&p| sector.contains(p)))
        .enumerate()
    {
        if sector.contains(*pos) {
            state.count_in_sector(index);
        } else {
            state.count_out_sector();
        }
    }

    state.extract(circle_surface)
}

#[derive(Clone, Debug)]
struct RingState {
    max_start: usize,
    max_len: usize,
    cur_start: usize,
    cur_len: usize,
}

impl RingState {
    #[inline]
    pub fn new() -> Self {
        Self {
            max_start: 0,
            max_len: 0,
            cur_start: 0,
            cur_len: 0,
        }
    }

    #[inline]
    pub fn is_on(&self) -> bool {
        self.cur_len > 0
    }

    // #[inline]
    // pub fn is_off(&self) -> bool {
    //     !self.is_on()
    // }

    #[inline]
    pub fn count_in_sector(&mut self, index: usize) {
        if self.is_on() {
            self.cur_len += 1;
        } else {
            self.cur_start = index;
            self.cur_len = 1;
        }
        // 每次在遇到扇区内的索引时试更新就足够了.
        self.try_update();
    }

    #[inline]
    pub fn count_out_sector(&mut self) {
        if self.is_on() {
            self.cur_len = 0;
        }
    }

    pub fn extract(&self, raw: &[Idx2d]) -> Vec<Idx2d> {
        raw.iter()
            .chain(raw)
            .skip(self.max_start)
            .take(self.max_len)
            .copied()
            .collect()
    }

    #[inline]
    fn try_update(&mut self) {
        if self.cur_len > self.max_len {
            (self.max_start, self.max_len) = (self.cur_start, self.cur_len);
        }
    }
}
