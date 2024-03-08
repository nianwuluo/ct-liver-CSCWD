use super::{idx3d_to_u16, ElemType};
use crate::{Idx3d, Idx3dU16};
use std::collections::{HashMap, HashSet};
use std::mem;

/// 实现提取 center roi 所需要维护的数据结构集合.
pub struct PhantomMemento {
    visited: HashSet<Idx3dU16>,
    data: HashMap<Idx3dU16, ElemType>,
    surf1: HashSet<Idx3dU16>,
    surf2: HashSet<Idx3dU16>,
}

impl PhantomMemento {
    pub fn new() -> Self {
        Self {
            visited: HashSet::with_capacity(4096),
            data: HashMap::with_capacity(4096),
            surf1: HashSet::with_capacity(1024),
            surf2: HashSet::with_capacity(1024),
        }
    }
    /// 为当前迭代缓存添加一个坐标. 返回值指示是否是插入了新值.
    #[inline]
    pub fn push_pos(&mut self, pos: &Idx3d) -> bool {
        self.surf1.insert(idx3d_to_u16(pos))
    }

    /// 获得所有即将要处理的肝脏下标 (新边缘集).
    ///
    /// 注意该操作在底层是 move 的, 因此性能开销较小.
    pub fn take_positions(&mut self) -> HashSet<Idx3dU16> {
        let prev_len = self.surf1.len();
        mem::replace(&mut self.surf1, HashSet::with_capacity(prev_len))
    }

    /// 为下次迭代缓存添加一个坐标. 返回值指示是否是插入了新值.
    #[inline]
    pub fn push_pos_next(&mut self, pos: &Idx3d) -> bool {
        self.surf2.insert(idx3d_to_u16(pos))
    }

    /// 进入下一步迭代.
    ///
    /// # 内部行为
    ///
    /// 清空 `self.surf1` 并与 `self.surf2` 交换.
    pub fn step(&mut self) {
        // 可能不需要, 因为 `take_positions()` 已经将 `self.surf1` 偷走.
        self.surf1.clear();

        mem::swap(&mut self.surf1, &mut self.surf2);
    }

    /// 设置某个体素索引的值为前景.
    #[inline]
    pub fn set_foreground(&mut self, pos: &Idx3d) {
        self.data.insert(idx3d_to_u16(pos), ElemType::Foreground);
    }

    /// 设置某个体素索引的值为背景.
    #[inline]
    pub fn set_background(&mut self, pos: &Idx3d) {
        self.data.insert(idx3d_to_u16(pos), ElemType::Background);
    }

    /// 判断某索引体素是否为前景. 越界时 panic.
    #[inline]
    pub fn is_foreground(&self, pos: &Idx3d) -> bool {
        !self.is_background(pos)
    }

    /// 判断某索引体素是否为背景. 越界时 panic.
    #[inline]
    pub fn is_background(&self, pos: &Idx3d) -> bool {
        self.get_val(pos).unwrap().is_background()
    }

    /// 获得某个体素索引的值. 若越界则返回 `None`.
    #[inline]
    pub fn get_val(&self, pos: &Idx3d) -> Option<&ElemType> {
        self.data.get(&idx3d_to_u16(pos))
    }

    /// 记录当前位置已被访问过.
    #[inline]
    pub fn set_visited(&mut self, pos: &Idx3d) {
        self.visited.insert(idx3d_to_u16(pos));
    }

    /// 判断某个位置是否已被访问过.
    #[inline]
    pub fn is_visited(&self, pos: &Idx3d) -> bool {
        self.visited.contains(&idx3d_to_u16(pos))
    }
}
