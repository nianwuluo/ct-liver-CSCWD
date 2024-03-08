//! [hrvoje 算法](https://ieeexplore.ieee.org/abstract/document/8564595/).

use super::{Profile, Runtime};
use ct_berry::consts::gray::*;
use ct_berry::prelude::*;
use std::collections::HashSet;

impl<'a> Runtime<'a> {
    /// hrvoje 算法分析.
    ///
    /// 注意该操作会修改原数据并消费自我 (consume `self`).
    /// 如有必要, 请提前做好备份.
    pub fn run_hrvoje(mut self, profile: &mut Profile) -> LabelSliceMut<'a> {
        if !self.unify_binary() {
            profile.count_trivial();
            return self.consume();
        }
        profile.count_target(true);

        // 前景集合, 时刻维护.
        let mut fg: HashSet<Idx2d> = self.liver_pos();

        // 获得粗略边缘
        let boundaries: Vec<Idx2d> = fg
            .iter()
            .copied()
            .filter(|p| self.is_n4_containing(*p, LITS_BACKGROUND))
            .collect();

        // 最终该值会成为腐蚀值; 目前记录为所有粗略边缘的值,
        // 后面会正确维护.
        let mut cnt = boundaries.len() as u64;

        for pos in boundaries.iter().copied() {
            self[pos] = LITS_BACKGROUND;
            fg.remove(&pos);
        }

        loop {
            // `bd` 的过滤条件是 hrvoje 算法的核心逻辑.
            let bd: Vec<Idx2d> = fg
                .iter()
                .copied()
                .filter(|p| self.n4_count(*p, LITS_LIVER) < 2)
                .collect();
            if bd.is_empty() {
                break;
            }
            cnt += bd.len() as u64;
            for pos in bd.iter().copied() {
                self[pos] = LITS_BACKGROUND;
                fg.remove(&pos);
            }
        }

        // 注意从上面的循环跳出后, 根据 hrvoje 算法的特性,
        // 可能生成多个孔洞. 需要考虑去除问题.
        let areas = self.areas_from_local(fg.iter().copied(), is_liver);
        if areas.is_empty() {
            profile.count_eroded(cnt);

            drop(fg);
            drop(boundaries);
            drop(areas);

            profile.target_elapsed();
            return self.consume();
        }
        let index = areas
            .iter()
            .enumerate()
            .max_by_key(|v| v.1.len())
            .unwrap()
            .0;
        for (_, v) in areas.iter().enumerate().filter(|(idx, _)| *idx != index) {
            cnt += v.len() as u64;
            v.iter().copied().for_each(|p| {
                self[p] = LITS_BACKGROUND;
                fg.remove(&p);
            });
        }

        for pos in fg.iter().copied() {
            let n4_pos = self.n4_positions(pos);
            for neigh in n4_pos.into_iter() {
                if self[neigh] == LITS_BACKGROUND {
                    self[neigh] = LITS_BOUNDARY;
                    cnt -= 1;
                }
            }
        }

        profile.count_eroded(cnt);

        // 为公平起见, 在计时前应当删除所有堆分配.
        drop(fg);
        drop(boundaries);
        drop(areas);

        profile.target_elapsed();

        self.consume()
    }
}
