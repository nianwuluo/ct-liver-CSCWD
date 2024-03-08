use super::{LabelMirror, ScanMirror};
use crate::consts::gray::*;
use crate::{Area2d, Areas2d, Idx2d, Predicate};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use ndarray::iter::{Iter, IterMut};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Ix2};
use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};
use std::io::{Read, Write};
use std::ops::{Index, IndexMut};

/// 不可变、借用的二维水平 CT 标签切片.
pub struct LabelSlice<'a> {
    /// 底层数据的轻量级视图, 借用于 [`crate::CtLabel`].
    ///
    /// 这里有意把代码写死为 `ArrayView` 降低灵活性, 但使结构的意图更加明确.
    data: ArrayView2<'a, u8>,
}

impl Index<Idx2d> for LabelSlice<'_> {
    type Output = u8;

    #[inline]
    fn index(&self, index: Idx2d) -> &Self::Output {
        &self.data[index]
    }
}

/// 可变、借用的二维水平 CT 标签切片.
pub struct LabelSliceMut<'a> {
    /// 底层数据的轻量级视图, 借用于 [`crate::CtLabel`].
    ///
    /// 这里有意把代码写死为 `ArrayViewMut` 降低灵活性, 但使结构的意图更加明确.
    data: ArrayViewMut2<'a, u8>,
}

/// 可变方法集合.
impl<'a> LabelSliceMut<'a> {
    /// 获得 **底层** 数据的一份可变 shallow copy.
    #[inline]
    pub fn array_view_mut(&mut self) -> ArrayViewMut2<u8> {
        self.data.view_mut()
    }

    /// 获取可以迭代并修改图像像素的迭代器.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, u8, Ix2> {
        self.data.iter_mut()
    }

    /// 获取给定位置 (高, 宽) 的像素值, 并可就地修改. 越界时返回 `None`.
    #[inline]
    pub fn get_mut(&mut self, pos: Idx2d) -> Option<&mut u8> {
        self.data.get_mut(pos)
    }

    /// 用 `mirror` 覆写原本 `self` 的内容.
    ///
    /// 如果 `mirror` 大小与 `self.len()` 不符, 则程序 panic.
    pub fn resume(&mut self, mirror: &LabelMirror) {
        assert_eq!(self.size(), mirror.0.len(), "镜像大小不符");
        for (r, w) in self.mirror().0.iter().zip(self.iter_mut()) {
            *w = *r;
        }
    }

    /// 将水平切片标注中值为 `old` 的像素全部替换为 `new`.
    ///
    /// 返回总共成功替换的个数.
    pub fn replace(&mut self, old: u8, new: u8) -> usize {
        let mut cnt = 0usize;
        self.array_view_mut()
            .iter_mut()
            .filter(|pix| **pix == old)
            .for_each(|p| {
                cnt += 1;
                *p = new;
            });
        cnt
    }

    /// 肝脏唯一化.
    ///
    /// 该函数先 **将所有肿瘤像素填充为肝脏像素**, 然后以 4-邻接规则实心化最大肝脏 (若有).
    /// 如果 **修改后的** 图片为全背景, 则返回 `false`; 否则返回 `true`.
    pub fn unify_binary(&mut self) -> bool {
        if !self.migrate_tumors() {
            return false;
        }
        // 运行到这里说明存在肝脏像素, 一定返回 true.
        let liver_areas = self.liver_areas();
        debug_assert!(!liver_areas.is_empty());

        self.non_max_filling(liver_areas, LITS_BACKGROUND);
        let bg_areas = self.background_areas();
        for area in bg_areas.iter() {
            if self.all_within(area) {
                self.fill_batch(area.iter().copied(), LITS_LIVER);
            }
        }
        true
    }

    /// 将图像中的背景空洞 (即连通背景区域中面积不是最大的那些)
    /// 填充为肝脏像素. 如果以此法修改了原图则返回 `true`,
    /// 否则返回 `false`.
    pub fn fill_background_hollow(&mut self) -> bool {
        let bg_areas = self.background_areas();
        let non_trivial = bg_areas.len() > 1;
        self.non_max_filling(bg_areas, LITS_LIVER);
        non_trivial
    }
}

impl Index<Idx2d> for LabelSliceMut<'_> {
    type Output = u8;

    #[inline]
    fn index(&self, index: Idx2d) -> &Self::Output {
        &self.data[index]
    }
}

/// label 不可变方法集合.
macro_rules! impl_label_slice_immut {
    ($life: lifetime, $slice: ty, $array: ty) => {
        /// 不可变方法集合.
        impl<$life> $slice {
            /// 直接初始化.
            #[inline]
            pub(crate) fn new(data: $array) -> Self {
                Self { data }
            }

            /// 获得 **底层** 数据的一份不可变 shallow copy.
            #[inline]
            pub fn array_view(&self) -> ArrayView2<u8> {
                self.data.view()
            }

            /// 获取可以迭代图像像素的迭代器.
            #[inline]
            pub fn iter(&self) -> Iter<'_, u8, Ix2> {
                self.data.iter()
            }

            /// 获取给定位置 (高, 宽) 的像素值. 越界时返回 `None`.
            #[inline]
            pub fn get(&self, pos: Idx2d) -> Option<&u8> {
                self.data.get(pos)
            }

            /// 该图是否为全背景图?
            #[inline]
            pub fn is_background(&self) -> bool {
                self.data.iter().copied().all(is_background)
            }

            /// 图像的分辨率 (高, 宽).
            #[inline]
            pub fn shape(&self) -> Idx2d {
                let &[h, w] = self.data.shape() else {
                    unreachable!()
                };
                (h, w)
            }

            /// 图像的像素个数.
            #[inline]
            pub fn size(&self) -> usize {
                let (h, w) = self.shape();
                h * w
            }

            /// 判断一个索引是否合法 (未越界).
            #[inline]
            pub fn check(&self, (h, w): Idx2d) -> bool {
                let (h_len, w_len) = self.shape();
                h < h_len && w < w_len
            }

            /// 统计图像中值为 `label` 的像素总个数.
            #[inline]
            pub fn count(&self, label: u8) -> usize {
                self.data.iter().filter(|&p| *p == label).count()
            }

            /// 将图像转化为行优先的序列化存储.
            pub fn as_row_major_vec(&self) -> Vec<u8> {
                let mut buf = Vec::with_capacity(self.size());
                buf.extend(self.iter());
                buf
            }

            /// 获得行优先存储的序列化数据.
            /// 当原始数据本身就是行优先格式时, 可以避免一次 deepcopy.
            pub fn as_row_major_slice(&self) -> Cow<[u8]> {
                if self.data.is_standard_layout() {
                    Cow::Borrowed(self.data.as_slice().unwrap())
                } else {
                    Cow::Owned(self.as_row_major_vec())
                }
            }

            /// 获取 CT 标签的基本统计信息.
            ///
            /// 统计信息格式为: \[背景像素数, 肝脏像素数, 肿瘤像素数\].
            /// 该操作不会统计任何其他像素信息.
            pub fn numeric_statistics(&self) -> [usize; 3] {
                let mut ans = [0; 3];
                for pixel in self.array_view().iter().filter(|p| **p <= 2) {
                    ans[*pixel as usize] += 1;
                }
                ans
            }

            /// 获取拥有所有权的镜像, 供以后可能的恢复.
            #[inline]
            pub fn mirror(&self) -> LabelMirror {
                self.into()
            }

            /// 获得一份不可变的 **本体** shallow copy.
            #[inline]
            pub fn shallow_copy(&self) -> LabelSlice {
                LabelSlice { data: self.array_view() }
            }

            /// 克隆自己, 获得一个拥有所有权的切片对象.
            pub fn to_owned(&self) -> OwnedLabelSlice {
                OwnedLabelSlice {
                    data: self.data.to_owned(),
                }
            }

            /// 获得图像的高.
            #[inline]
            pub fn height(&self) -> usize {
                self.shape().0
            }

            /// 获得图像的宽.
            #[inline]
            pub fn width(&self) -> usize {
                self.shape().1
            }

            /// 判断一个索引是否位于图像的边缘.
            #[inline]
            pub fn is_at_border(&self, (h, w): Idx2d) -> bool {
                h == 0
                    || h.saturating_add(1) == self.height()
                    || w == 0
                    || w.saturating_add(1) == self.width()
            }

            /// 获得 `pos` 的 4-邻域像素索引. 保证返回的索引都不越界.
            pub fn n4_positions(&self, pos: Idx2d) -> Vec<Idx2d> {
                crate::eight::neighbour4(pos)
                    .into_iter()
                    .filter(|p| self.check(*p))
                    .collect()
            }

            /// 获得 `pos` 的 8-邻域像素索引. 保证返回的索引都不越界.
            pub fn n8_positions(&self, pos: Idx2d) -> Vec<Idx2d> {
                crate::eight::neighbour8(pos)
                    .into_iter()
                    .filter(|p| self.check(*p))
                    .collect()
            }

            /// 判断 `pos` 的 4-邻域是否含有 `target`.
            #[inline]
            pub fn is_n4_containing(&self, pos: Idx2d, target: u8) -> bool {
                self.is_n4_having(pos, |pix| pix == target)
            }

            /// 判断 `pos` 的 4-邻域是否含有背景像素 `LITS_BACKGROUND`.
            #[inline]
            pub fn is_n4_containing_background(&self, pos: Idx2d) -> bool {
                self.is_n4_having(pos, is_background)
            }

            /// 判断 `pos` 的 4-邻域是否含有背景像素 `LITS_LIVER`.
            #[inline]
            pub fn is_n4_containing_liver(&self, pos: Idx2d) -> bool {
                self.is_n4_having(pos, is_liver)
            }

            /// 判断 `pos` 的 4-邻域是否含有背景像素 `LITS_TUMOR`.
            #[inline]
            pub fn is_n4_containing_tumor(&self, pos: Idx2d) -> bool {
                self.is_n4_having(pos, is_tumor)
            }

            /// 判断 `(h, w)` 的 4-邻域是否由满足谓词 `pred` 的像素.
            pub fn is_n4_having(&self, (h, w): Idx2d, mut pred: impl FnMut(u8) -> bool) -> bool {
                matches!(self.get((h.wrapping_sub(1), w)), Some(&v) if pred(v))
                    || matches!(self.get((h.saturating_add(1), w)), Some(&v) if pred(v))
                    || matches!(self.get((h, w.wrapping_sub(1))), Some(&v) if pred(v))
                    || matches!(self.get((h, w.saturating_add(1))), Some(&v) if pred(v))
            }

            /// 以行优先规则, 获取能迭代图像所有索引的迭代器.
            #[inline]
            pub fn pos_iter(&self) -> impl Iterator<Item = Idx2d> {
                super::iter::PosIter::new(self.shape())
            }

            /// 以行优先规则, 获取能迭代图像所有 `(索引, 像素值)` 的迭代器.
            #[inline]
            pub fn indexed_iter(&self) -> impl Iterator<Item = (Idx2d, &u8)> {
                self.data.indexed_iter()
            }

            /// 以 4-邻域为腐蚀核, 获取前景中心位置. 如果图片为全背景则返回 `None`.
            /// 前景的规则由 `is_foreground` 指定. 内部实现保证每次运行产生同样的结果.
            ///
            /// # 注意
            ///
            /// `is_foreground(LITS_BACKGROUND)` 必须返回 `false`,
            /// 否则程序行为未定义.
            fn n4_center(&self, is_foreground: Predicate) -> Option<Idx2d> {
                let mut last: Idx2d = (usize::MAX, usize::MAX);
                let mut q: VecDeque<Idx2d> = self
                    .array_view()
                    .indexed_iter()
                    .filter_map(|(pos, &pix)| {
                        (is_foreground(pix) && self.is_n4_containing(pos, LITS_BACKGROUND)).then_some(pos)
                    })
                    .collect();

                if q.is_empty() {
                    return None;
                }
                let mut vis = HashSet::with_capacity(q.len());
                while !q.is_empty() {
                    let cur = q.pop_front().unwrap();
                    if vis.contains(&cur) {
                        continue;
                    }
                    vis.insert(cur);
                    q.extend(
                        self.n4_positions(cur)
                            .into_iter()
                            .filter(|neigh| is_foreground(self[*neigh]) && !vis.contains(neigh)),
                    );
                    last = cur;
                }

                debug_assert_ne!(last.0, usize::MAX);
                Some(last)
            }

            /// 以 4-邻域为腐蚀核, 获取肝脏中心位置. 如果图片为全背景则返回 `None`.
            /// 内部实现保证每次运行产生同样的结果.
            #[inline]
            pub fn n4_liver_center(&self) -> Option<Idx2d> {
                self.n4_center(is_liver)
            }

            /// 以 4-邻域为腐蚀核, 获取肝脏-肿瘤实体中心位置. 如果图片为全背景则返回
            /// `None`. 内部实现保证每次运行产生同样的结果.
            #[inline]
            pub fn n4_lt_center(&self) -> Option<Idx2d> {
                self.n4_center(is_liver_or_tumor)
            }


            /// 判断图像上是否有肿瘤 [`LITS_TUMOR`] 像素.
            #[inline]
            pub fn has_tumor(&self) -> bool {
                self.iter().any(|c| is_tumor(*c))
            }

            /// 判断图像上是否有肿瘤 [`LITS_LIVER`] 像素.
            #[inline]
            pub fn has_liver(&self) -> bool {
                self.iter().any(|c| is_liver(*c))
            }

            /// 按照 4-相邻规则获取所有区域. 两个像素 `p1` 和 `p2` 属于同一个区域,
            /// 当且仅当存在一条从 `p1` 到 `p2` 的 4-相邻路径, 且路径上的所有像素
            /// (包括 `p1` 和 `p2`) 都满足谓词 `pred`.
            pub fn areas(&self, pred: Predicate) -> Areas2d {
                self.areas_from_local(self.pos_iter(), pred)
            }

            /// 按照 4-相邻原则获得图像中所有背景区域.
            #[inline]
            pub fn background_areas(&self) -> Areas2d {
                self.areas(is_background)
            }

            /// 按照 4-相邻原则获得图像中所有肝脏区域.
            #[inline]
            pub fn liver_areas(&self) -> Areas2d {
                self.areas(is_liver)
            }

            /// 按照 4-相邻原则获得图像中所有肿瘤区域.
            #[inline]
            pub fn tumor_areas(&self) -> Areas2d {
                self.areas(is_tumor)
            }

            /// 按照 4-相邻原则获得图像中所有肝脏/肿瘤区域
            /// (两种像素被视为等价, 即允许在一个 `Area` 中混合肝脏和肿瘤像素).
            #[inline]
            pub fn lt_areas(&self) -> Areas2d {
                self.areas(is_liver_or_tumor)
            }

            /// 获取所有肝脏像素的索引.
            pub fn liver_pos<B: FromIterator<Idx2d>>(&self) -> B {
                FromIterator::from_iter(
                    self.array_view()
                        .indexed_iter()
                        .filter_map(|(pos, pixel)| is_liver(*pixel).then_some(pos))
                )
            }

            /// 按照 4-相邻规则获取所有区域, 但区域范围由 `it` 指定.
            /// 两个像素 `p1` 和 `p2` 属于同一个区域, 当且仅当存在一条从 `p1` 到
            /// `p2` 的 4-相邻路径, 且路径上的所有像素 (包括 `p1` 和 `p2`)
            /// 都满足谓词 `pred`.
            pub fn areas_from_local<I: IntoIterator<Item = Idx2d>>(
                &self,
                it: I,
                pred: Predicate,
            ) -> Areas2d {
                let mut ans = Areas2d::with_capacity(1);
                let mut bfs_q = VecDeque::with_capacity(4);
                let mut set = HashSet::with_capacity(16);

                for pos in it.into_iter() {
                    if set.contains(&pos) || !pred(self[pos]) {
                        continue;
                    }
                    bfs_q.push_back(pos);
                    let mut this_area = Area2d::with_capacity(1);
                    while !bfs_q.is_empty() {
                        let cur_pos = bfs_q.pop_front().unwrap();
                        if set.contains(&cur_pos) {
                            continue;
                        }
                        set.insert(cur_pos);
                        this_area.push(cur_pos);

                        // bfs
                        let (cur_h, cur_w) = cur_pos;
                        if cur_h > 0
                            && pred(self[(cur_h - 1, cur_w)])
                            && !set.contains(&(cur_h - 1, cur_w))
                        {
                            bfs_q.push_back((cur_h - 1, cur_w));
                        }
                        if cur_h.wrapping_add(1) < self.height()
                            && pred(self[(cur_h + 1, cur_w)])
                            && !set.contains(&(cur_h + 1, cur_w))
                        {
                            bfs_q.push_back((cur_h + 1, cur_w));
                        }
                        if cur_w > 0
                            && pred(self[(cur_h, cur_w - 1)])
                            && !set.contains(&(cur_h, cur_w - 1))
                        {
                            bfs_q.push_back((cur_h, cur_w - 1));
                        }
                        if cur_w.wrapping_add(1) < self.width()
                            && pred(self[(cur_h, cur_w + 1)])
                            && !set.contains(&(cur_h, cur_w + 1))
                        {
                            bfs_q.push_back((cur_h, cur_w + 1));
                        }
                    }
                    ans.push(this_area);
                }
                ans
            }
        }
    };
}
impl_label_slice_immut!('a, LabelSlice<'a>, ArrayView2<'a, u8>);
impl_label_slice_immut!('a, LabelSliceMut<'a>, ArrayViewMut2<'a, u8>);

impl IndexMut<Idx2d> for LabelSliceMut<'_> {
    #[inline]
    fn index_mut(&mut self, index: Idx2d) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 拥有所有权的二维水平 CT 标签切片.
///
/// `OwnedLabelSlice` 仅提供到 `LabelSlice` 和 `LabelSliceMut`
/// 的轻量转换和底层数据移动, 不提供任何其它方法.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct OwnedLabelSlice {
    data: Array2<u8>,
}

impl OwnedLabelSlice {
    /// 获得不可变切片引用.
    #[inline]
    pub fn as_immut(&self) -> LabelSlice<'_> {
        LabelSlice::new(self.data.view())
    }

    /// 获得可变切片引用.
    #[inline]
    pub fn as_mutable(&mut self) -> LabelSliceMut<'_> {
        LabelSliceMut::new(self.data.view_mut())
    }

    /// 直接获得底层数据.
    #[inline]
    pub fn into_raw(self) -> Array2<u8> {
        self.data
    }
}

impl OwnedLabelSlice {
    /// 压缩数据.
    pub fn compress(&self) -> CompactLabelSlice {
        let data = self.as_immut();
        let buf = data.as_row_major_slice();
        let mut e = ZlibEncoder::new(Vec::with_capacity(8), Compression::best());
        e.write_all(buf.as_ref()).expect("Compression error");
        let sh = data.shape();
        CompactLabelSlice {
            buf: e.finish().expect("Compression error"),
            sh,
        }
    }
}

/// 压缩存储的 `OwnedLabelSlice`; 不透明类型.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompactLabelSlice {
    /// 压缩的不透明字节流.
    buf: Vec<u8>,

    /// 形状.
    sh: Idx2d,
}

impl CompactLabelSlice {
    /// 解压缩数据.
    pub fn decompress(self) -> OwnedLabelSlice {
        let Self { buf, sh: (h, w) } = self;
        let mut d = ZlibDecoder::new(buf.as_slice());
        let mut buf = Vec::with_capacity(h * w);
        d.read_to_end(&mut buf).expect("Decompression error");
        debug_assert_eq!(buf.len(), h * w);
        let data = Array2::<u8>::from_shape_vec((h, w), buf).unwrap();
        OwnedLabelSlice { data }
    }
}

/// 不可变、借用的二维水平 CT 扫描切片.
pub struct ScanSlice<'a> {
    /// 底层数据的轻量级视图, 借用于 [`crate::CtScan`].
    ///
    /// 这里有意把代码写死为 `ArrayView` 降低灵活性, 但使结构的意图更加明确.
    data: ArrayView2<'a, f32>,
}

impl Index<Idx2d> for ScanSlice<'_> {
    type Output = f32;

    #[inline]
    fn index(&self, index: Idx2d) -> &Self::Output {
        &self.data[index]
    }
}

/// 可变、借用的二维水平 CT 扫描切片.
pub struct ScanSliceMut<'a> {
    /// 底层数据的轻量级视图, 借用于 [`crate::CtScan`].
    ///
    /// 这里有意把代码写死为 `ArrayViewMut` 降低灵活性, 但使结构的意图更加明确.
    data: ArrayViewMut2<'a, f32>,
}

/// 可变方法集合.
impl<'a> ScanSliceMut<'a> {
    /// 获得数据的一份可变 shallow copy.
    #[inline]
    pub fn data_mut(&mut self) -> ArrayViewMut2<f32> {
        self.data.view_mut()
    }

    /// 获取可以迭代并修改图像像素的迭代器.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, f32, Ix2> {
        self.data.iter_mut()
    }

    /// 获取给定位置 (高, 宽) 的像素值, 并可就地修改. 越界时返回 `None`.
    #[inline]
    pub fn get_mut(&mut self, pos: Idx2d) -> Option<&mut f32> {
        self.data.get_mut(pos)
    }

    /// 用 `mirror` 覆写原本 `self` 的内容.
    ///
    /// 如果 `mirror` 大小与 `self.len()` 不符, 则程序 panic.
    pub fn resume(&mut self, mirror: &ScanMirror) {
        assert_eq!(self.size(), mirror.0.len(), "镜像大小不符");
        for (r, w) in self.mirror().0.iter().zip(self.iter_mut()) {
            *w = *r;
        }
    }
}

impl Index<Idx2d> for ScanSliceMut<'_> {
    type Output = f32;

    #[inline]
    fn index(&self, index: Idx2d) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<Idx2d> for ScanSliceMut<'_> {
    #[inline]
    fn index_mut(&mut self, index: Idx2d) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// scan 不可变方法集合.
macro_rules! impl_scan_slice_immut {
    ($life: lifetime, $scan: ty, $array: ty) => {
        /// 不可变方法集合.
        impl<$life> $scan {
            /// 直接初始化.
            #[inline]
            pub(crate) fn new(data: $array) -> Self {
                Self { data }
            }

            /// 获得数据的一份不可变 shallow copy.
            #[inline]
            pub fn data(&self) -> ArrayView2<f32> {
                self.data.view()
            }

            /// 获取可以迭代图像像素的迭代器.
            #[inline]
            pub fn iter(&self) -> Iter<'_, f32, Ix2> {
                self.data.iter()
            }

            /// 获取给定位置 (高, 宽) 的像素值. 越界时返回 `None`.
            #[inline]
            pub fn get(&self, pos: Idx2d) -> Option<&f32> {
                self.data.get(pos)
            }

            /// 图像的分辨率 (高, 宽).
            #[inline]
            pub fn shape(&self) -> Idx2d {
                let &[h, w] = self.data.shape() else {
                    unreachable!()
                };
                (h, w)
            }

            /// 图像的像素个数.
            #[inline]
            pub fn size(&self) -> usize {
                let (h, w) = self.shape();
                h * w
            }

            /// 获取拥有所有权的镜像, 供以后可能的恢复.
            #[inline]
            pub fn mirror(&self) -> ScanMirror {
                self.into()
            }

            /// 克隆自己, 获得一个拥有所有权的切片对象.
            pub fn to_owned(&self) -> OwnedScanSlice {
                OwnedScanSlice {
                    data: self.data.to_owned(),
                }
            }

            /// 以行优先规则, 获取能迭代图像所有 `(索引, CT HU 值)` 的迭代器.
            #[inline]
            pub fn indexed_iter(&self) -> impl Iterator<Item = (Idx2d, &f32)> {
                self.data.indexed_iter()
            }
        }
    };
}

impl_scan_slice_immut!('a, ScanSlice<'a>, ArrayView2<'a, f32>);
impl_scan_slice_immut!('a, ScanSliceMut<'a>, ArrayViewMut2<'a, f32>);

/// 拥有所有权的二维水平 CT 扫描切片.
///
/// `OwnedScanSlice` 仅提供到 `ScanSlice` 和 `ScanSliceMut`
/// 的轻量转换和底层数据移动, 不提供任何其它方法.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OwnedScanSlice {
    data: Array2<f32>,
}

impl OwnedScanSlice {
    /// 获得不可变切片引用.
    #[inline]
    pub fn as_immutable(&self) -> ScanSlice<'_> {
        ScanSlice::new(self.data.view())
    }

    /// 获得可变切片引用.
    #[inline]
    pub fn as_mutable(&mut self) -> ScanSliceMut<'_> {
        ScanSliceMut::new(self.data.view_mut())
    }

    /// 直接获得底层数据.
    #[inline]
    pub fn into_raw(self) -> Array2<f32> {
        self.data
    }
}
