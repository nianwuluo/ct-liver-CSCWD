//! 我提出的算法.

use super::{Profile, Runtime};
use ct_berry::consts::gray::*;
use ct_berry::prelude::*;
use std::collections::HashSet;
use std::mem;

impl<'a> Runtime<'a> {
    /// 自研算法分析.
    ///
    /// 注意该操作会修改原数据并消费自我 (consume `self`).
    /// 如有必要, 请提前做好备份.
    pub fn run_mulberry(mut self, profile: &mut Profile) -> LabelSliceMut<'a> {
        if !self.unify_binary() {
            profile.count_trivial();
            return self.consume();
        }
        // 运行到这里, 就存在且仅存在一个 4-邻域连接的肝脏实心体了.
        profile.count_target(true);

        // 肝脏像素
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
        let mut eroded = 0u64;
        for pos in e_set {
            if !self.is_n4_containing(pos, LITS_LIVER) {
                eroded += 1;
                g_max0.remove(&pos);
            }
        }

        // 0-1 4-邻接分组
        let mut s_g01 = self.areas_from_local(g_max0.iter().copied(), is_liver_or_boundary);
        if s_g01.is_empty() {
            profile.count_eroded(eroded);
            return self.consume();
        }

        let index = s_g01
            .iter()
            .enumerate()
            .max_by_key(|v| v.1.len())
            .unwrap()
            .0;

        // 非最大区域以背景填充, 只留一个 0-1 4-邻接分组.
        for (_, v) in s_g01.iter().enumerate().filter(|(idx, _)| *idx != index) {
            profile.count_eroded(v.len() as u64);
            self.fill_batch(v.iter().copied(), LITS_BACKGROUND);
            // v.iter().copied().for_each(|p| self[p] = LITS_BACKGROUND);
        }
        let s_g01_one = mem::take(&mut s_g01[index]);

        // 0-1 4-邻接分组的所有肝脏区域, 只留一个最大的, 其它涂为边缘.
        // note: 这一步可能会腐蚀掉更多像素, 因此不是最优.
        let s_g0_1 = self.areas_from_local(s_g01_one.iter().copied(), is_liver);
        self.non_max_filling(s_g0_1, LITS_BOUNDARY);

        // 4-相邻没包含肝脏的边缘涂为背景
        let mut local_eroded = 0;
        for pos in s_g01_one.iter().copied() {
            if self[pos] == LITS_BOUNDARY && !self.is_n4_containing(pos, LITS_LIVER) {
                self[pos] = LITS_BACKGROUND;
                local_eroded += 1;
            }
        }
        profile.count_eroded(local_eroded);

        // 为公平起见, 在计时前应当删除所有堆分配.
        drop(g_max0);
        drop(s_g01);
        drop(s_g01_one);

        profile.target_elapsed();

        self.consume()
    }
}
