use crate::Idx2d;

/// 行优先索引迭代器.
///
/// 虽然如下函数也能实现相同的功能:
///
/// ```
/// type Idx2d = (usize, usize);
///
/// fn pos_iter_auto((h, w): Idx2d) -> impl Iterator<Item = Idx2d> {
///     (0..h).flat_map(move |first| (0..w).map(move |second| (first, second)))
/// }
///
/// // ...
/// ```
///
/// 但经测试, 该迭代器对象占用的空间是手写 `PosIter` 的三倍. 因此为性能考虑,
/// 我们保留该结构.
#[derive(Debug)]
pub struct PosIter {
    cur_h: usize,
    cur_w: usize,
    h: usize,
    w: usize,
}

impl PosIter {
    #[inline]
    pub fn new((h, w): Idx2d) -> Self {
        Self {
            cur_h: 0,
            cur_w: 0,
            h,
            w,
        }
    }
}

impl Iterator for PosIter {
    type Item = Idx2d;

    fn next(&mut self) -> Option<Self::Item> {
        if self.h == 0 || self.w == 0 || self.cur_h == self.h {
            return None;
        }
        let ret_pos = (self.cur_h, self.cur_w);
        if self.cur_w + 1 == self.w {
            self.cur_w = 0;
            self.cur_h += 1;
        } else {
            self.cur_w += 1;
        }
        Some(ret_pos)
    }
}

/// 该测试已足够覆盖所有情况, 不用变更.
#[cfg(test)]
mod completeness_tests {
    use super::PosIter;
    use crate::Idx2d;

    fn pos_iter_builtin((h, w): Idx2d) -> impl Iterator<Item = Idx2d> {
        (0..h).flat_map(move |first| (0..w).map(move |second| (first, second)))
    }

    #[test]
    fn test_builtin_iter_size_larger() {
        use std::mem::size_of_val as sizeof;

        let tup = (1, 1);
        assert!(sizeof(&pos_iter_builtin(tup)) > sizeof(&PosIter::new(tup)));
    }

    #[test]
    fn test_pos_iter() {
        // 这几个基本例子足以证明正确性了.
        for i in 0..=4 {
            for j in 0..=4 {
                let tup = (i, j);
                assert!(Iterator::eq(pos_iter_builtin(tup), PosIter::new(tup)));
                // assert_eq!(pos_iter_builtin(tup), PosIter::new(tup));
            }
        }
    }
}
