mod canny;
mod hrvoje;
mod mulberry;
mod profile;
mod suzuki;

use ct_berry::prelude::*;
use opencv::core::Mat;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use utils::loader;

pub use profile::Profile;

type Area2d = Vec<Idx2d>;
type Areas2d = Vec<Area2d>;

/// 预处理算法/研究算法的核心操作集.
pub struct Runtime<'a> {
    data: LabelSliceMut<'a>,
}

/// 对 `Runtime` 的索引访问本质上是对 `label` 的访问.
impl<'a> Deref for Runtime<'a> {
    type Target = LabelSliceMut<'a>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// 对 `Runtime` 的索引访问本质上是对 `label` 的访问.
impl<'a> DerefMut for Runtime<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a> Runtime<'a> {
    /// 初始化 `Runtime`.
    #[inline]
    pub fn new(data: LabelSliceMut<'a>) -> Self {
        Self { data }
    }

    /// 消费自我, 获得底层数据.
    #[inline]
    pub fn consume(self) -> LabelSliceMut<'a> {
        self.data
    }

    /// 从 `self.dataset` 创建一个 owned 的 opencv matrix.
    ///
    /// 注意该函数存在效率方面的问题, 仅用于算法时间比较中.
    pub(crate) fn make_owned_opencv_matrix(&self) -> Mat {
        let (h, w) = self.shape();
        let contiguous = self.data.as_row_major_slice();
        Mat::from_slice_rows_cols(contiguous.as_ref(), h, w).unwrap()
    }

    /// 判断 `pos` 的 4-邻域中像素值为 `label` 的个数.
    pub(crate) fn n4_count(&self, pos: Idx2d, label: u8) -> usize {
        neighbour4(pos)
            .into_iter()
            .filter_map(|p| self.get(p).copied())
            .filter(|p| *p == label)
            .count()
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

    /// 将 `it` 中的每个索引对应的像素改为 `new`.
    pub(crate) fn fill_batch<I: IntoIterator<Item = Idx2d>>(&mut self, it: I, new: u8) {
        for pos in it.into_iter() {
            self[pos] = new;
        }
    }
}

/// 获得 `(h, w)` 的 4-邻居索引. 不检查越界.
#[inline]
fn neighbour4((h, w): Idx2d) -> [Idx2d; 4] {
    [
        (h.wrapping_sub(1), w),
        (h.saturating_add(1), w),
        (h, w.wrapping_sub(1)),
        (h, w.saturating_add(1)),
    ]
}

pub fn canny(p: &Path) -> Profile {
    let mut profile = Profile::new();
    for (nii_idx, labels) in loader::label_loader(p) {
        let mut label = labels.unwrap();
        println!("Canny: file {nii_idx}...");
        for sli in label.slice_iter_mut() {
            Runtime::new(sli).run_canny(&mut profile);
        }
    }
    profile.finish()
}

pub fn suzuki(p: &Path) -> Profile {
    let mut profile = Profile::new();
    for (nii_idx, labels) in loader::label_loader(p) {
        let mut label = labels.unwrap();
        println!("Suzuki: file {nii_idx}...");
        for sli in label.slice_iter_mut() {
            Runtime::new(sli).run_suzuki85(&mut profile);
        }
    }
    profile.finish()
}

pub fn hrvoje(p: &Path) -> Profile {
    let mut profile = Profile::new();
    for (nii_idx, labels) in loader::label_loader(p) {
        let mut label = labels.unwrap();
        println!("Hrvoje: file {nii_idx}...");
        for sli in label.slice_iter_mut() {
            Runtime::new(sli).run_hrvoje(&mut profile);
        }
    }
    profile.finish()
}

pub fn mulberry(p: &Path) -> Profile {
    let mut profile = Profile::new();
    for (nii_idx, labels) in loader::label_loader(p) {
        let mut label = labels.unwrap();
        println!("Mulberry: file {nii_idx}...");
        for sli in label.slice_iter_mut() {
            Runtime::new(sli).run_mulberry(&mut profile);
        }
    }
    profile.finish()
}
