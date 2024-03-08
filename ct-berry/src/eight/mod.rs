//! 邻域相关的边缘提取算法操作.

mod core;

use crate::Idx2d;

/// 获得 `(h, w)` 的 4-邻居索引. 不检查越界.
#[inline]
pub(crate) fn neighbour4((h, w): Idx2d) -> [Idx2d; 4] {
    [
        (h.wrapping_sub(1), w),
        (h.saturating_add(1), w),
        (h, w.wrapping_sub(1)),
        (h, w.saturating_add(1)),
    ]
}

/// 获得 `(h, w)` 的 8-邻居索引. 不检查越界.
#[inline]
pub(crate) fn neighbour8((h, w): Idx2d) -> [Idx2d; 8] {
    [
        (h.wrapping_sub(1), w.wrapping_sub(1)),
        (h.wrapping_sub(1), w),
        (h.wrapping_sub(1), w.saturating_add(1)),
        (h, w.wrapping_sub(1)),
        (h, w.saturating_add(1)),
        (h.saturating_add(1), w.wrapping_sub(1)),
        (h.saturating_add(1), w),
        (h.saturating_add(1), w.saturating_add(1)),
    ]
}
