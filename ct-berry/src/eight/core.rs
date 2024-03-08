use crate::consts::gray::*;
use crate::{Area2d, Areas2d, Idx2d, LabelSliceMut};
use std::collections::HashSet;

/// ours (mulberry) 算法实现块.
impl<'a> LabelSliceMut<'a> {
    /// 运行邻域提取算法. 返回首尾相连的 8-邻域边缘.
    ///
    /// # 注意
    ///
    /// 1. 需要保证切片中的像素值只有 `LITS_BACKGROUND`, `LITS_LIVER`,
    ///   `LITS_TUMOR` (分别为 0, 1, 2), 否则程序行为未定义.
    /// 2. 算法会修改原图, 保留最后实心肝脏对象的轮廓 (`LITS_BOUNDARY`)
    ///    和内部 (`LITS_LIVER`).
    pub fn mulberry(&mut self) -> Vec<Idx2d> {
        if !self.unify_binary() {
            return vec![];
        }

        // 肝脏像素.
        let mut g_max0: HashSet<Idx2d> = HashSet::with_capacity(64);

        // 与背景 4-相邻的所有肝脏像素
        let mut e_set: Vec<Idx2d> = Vec::with_capacity(64);

        for pos in self.pos_iter() {
            if self[pos] == LITS_LIVER {
                g_max0.insert(pos);
                if self.is_n4_containing(pos, LITS_BACKGROUND) {
                    e_set.push(pos);
                }
            }
        }

        // 将 4-相邻包含背景的肝脏, 涂为边缘
        self.fill_batch(e_set.iter().copied(), LITS_BOUNDARY);

        // 将 4-相邻没有包含肝脏的边缘, 涂为背景
        e_set
            .iter()
            .filter(|pos| !self.is_n4_containing(**pos, LITS_LIVER))
            .for_each(|pos| {
                g_max0.remove(pos);
            });

        // 0-1 4-邻接分组
        let mut s_g01 = self.areas_from_local(g_max0.iter().copied(), is_liver_or_boundary);

        // 最大索引
        let index = match s_g01.iter().enumerate().max_by_key(|v| v.1.len()) {
            None => return vec![],
            Some((idx, _)) => idx,
        };

        // 非最大区域以背景填充, 只留一个 0-1 4-邻接分组.
        s_g01
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != index)
            .for_each(|(_, v)| {
                self.fill_batch(v.iter().copied(), LITS_BACKGROUND);
            });

        // 目前唯一的 0-1 4-相邻区域
        let s_g01_one = std::mem::take(&mut s_g01[index]);

        // 0-1 4-邻接分组的所有肝脏区域, 只留一个最大的, 其它涂为边缘.
        // note: 这一步可能会腐蚀掉更多像素, 因此不是最优.
        let s_g0_1 = self.areas_from_local(s_g01_one.iter().copied(), is_liver);
        self.non_max_filling(s_g0_1, LITS_BOUNDARY);

        // 4-相邻没包含肝脏的边缘涂为背景
        let mut edge_seed = (usize::MAX, usize::MAX);
        for pos in s_g01_one.iter().copied() {
            if self[pos] == LITS_BOUNDARY {
                if self.is_n4_containing(pos, LITS_LIVER) {
                    edge_seed = pos;
                } else {
                    self[pos] = LITS_BACKGROUND;
                }
            }
        }

        self.get_surface_curve(edge_seed)
    }

    /// 从 `seed` 出发, 获取严格的 8-相邻稀疏边缘曲线.
    fn get_surface_curve(&self, seed: Idx2d) -> Vec<Idx2d> {
        if seed.0 == usize::MAX {
            return vec![];
        }

        debug_assert_eq!(self[seed], LITS_BOUNDARY);
        let mut last = seed;
        let mut last_last = (usize::MAX, usize::MAX);
        let mut ans = Vec::with_capacity(4);
        ans.push(seed);

        loop {
            debug_assert_eq!(
                2,
                self.n8_positions(last)
                    .iter()
                    .filter(|p| is_boundary(self[**p]))
                    .count(),
            );
            match self
                .n8_positions(last)
                .iter()
                .find(|&&p| is_boundary(self[p]) && p != last_last)
            {
                Some(&then) if then != seed => {
                    ans.push(then);
                    last_last = last;
                    last = then;
                }
                None => unreachable!(),
                _ => break,
            }
        }
        debug_assert!(ans.len() >= 3);
        fn is_n8_neighbouring((a, b): Idx2d, (c, d): Idx2d) -> bool {
            matches!((a.abs_diff(c), b.abs_diff(d)), (1, 0) | (0, 1) | (1, 1))
        }
        debug_assert!(is_n8_neighbouring(
            *ans.first().unwrap(),
            *ans.last().unwrap(),
        ));
        // debug_assert_eq!(self.count(LITS_BOUNDARY), ans.len());
        ans
    }

    /// 获取 `groups` 中第一个最大的区域并返回. 同时将其它区域都填充为 `fill_with`.
    ///
    /// 如果 `groups` 为空, 则返回 `None`.
    pub(crate) fn non_max_filling(&mut self, mut groups: Areas2d, fill_with: u8) -> Option<Area2d> {
        let index = groups.iter().enumerate().max_by_key(|v| v.1.len())?.0;
        for (_, area) in groups.iter().enumerate().filter(|(idx, _)| *idx != index) {
            area.iter().copied().for_each(|p| self[p] = fill_with);
        }
        Some(std::mem::take(&mut groups[index]))
    }

    /// 消除肿瘤像素 (若有) 并将其修改为肝脏像素.
    ///
    /// 如果 **修改前的原图** 为全背景图则返回 `false`, 否则返回 `true`.
    pub(crate) fn migrate_tumors(&mut self) -> bool {
        let mut all_bg = true;
        for pix in self.iter_mut() {
            if is_tumor(*pix) {
                *pix = LITS_LIVER;
            }
            if !is_tumor(*pix) {
                all_bg = false;
            }
        }
        !all_bg
    }

    /// 将 `it` 中的每个索引对应的像素改为 `new`.
    pub(crate) fn fill_batch<I: IntoIterator<Item = Idx2d>>(&mut self, it: I, new: u8) {
        for pos in it.into_iter() {
            self[pos] = new;
        }
    }

    /// 判断 `positions` 的索引是否全部都在图像的内部.
    #[inline]
    pub(crate) fn all_within(&self, positions: &[Idx2d]) -> bool {
        positions.iter().all(|p| !self.is_at_border(*p))
    }
}
