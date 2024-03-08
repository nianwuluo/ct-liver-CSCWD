use std::ops::{Index, IndexMut};
use std::path::Path;

use ndarray::{Array3, ArrayView, ArrayViewMut, Axis, Ix3};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};

use crate::consts::gray::*;
use crate::{Idx2d, Idx3d, Predicate};

pub mod morph_3d;
pub mod sector;
pub mod slice;
pub mod window;

use sector::{InitLlsPatternError, LlsSectorPattern};

pub use slice::{
    CompactLabelSlice, ImgWriteRaw, ImgWriteVis, LabelSlice, LabelSliceMut, OwnedLabelSlice,
    OwnedScanSlice, ScanSlice, ScanSliceMut,
};

#[cfg(feature = "plot")]
pub use slice::ImgDisplay;

pub use window::CtWindow;

/// `NiftiHeader` 是栈上大对象, 移动该对象的开销很可观.
/// 因此我们将其分配到堆上.
type BoxedHeader = Box<NiftiHeader>;

/// nii 格式 3D CT 扫描, 包括 header 和 CT 扫描 (HU). HU 值以 `f32` 保存.
#[derive(Debug, Clone)]
pub struct CtScan {
    header: BoxedHeader,
    data: Array3<f32>,
}

/// 将 (W, H, z) 转换成 (z, H, W). 以后均按照该模式访问.
#[inline]
fn get_shape_from_header(h: &NiftiHeader) -> Idx3d {
    // [W, H, z]. 体素个数数组.
    let [_, w, h, z, ..] = h.dim;
    (z as usize, h as usize, w as usize)
}

/// 3D CT nii 文件 header 的共用属性和部分通用操作.
pub trait NiftiHeaderAttr {
    /// 获取 header 部分.
    fn header(&self) -> &NiftiHeader;

    /// 获取数据形状大小.
    #[inline]
    fn shape(&self) -> Idx3d {
        get_shape_from_header(self.header())
    }

    /// 获取数据水平切片形状大小.
    #[inline]
    fn slice_shape(&self) -> Idx2d {
        let (_, h, w) = self.shape();
        (h, w)
    }

    /// 获取水平切片个数.
    #[inline]
    fn len_z(&self) -> usize {
        self.shape().0
    }

    /// 获取数据体素个数.
    #[inline]
    fn size(&self) -> usize {
        let (z, h, w) = self.shape();
        z * h * w
    }

    /// 检查索引是否合法.
    #[inline]
    fn check(&self, (z0, h0, w0): &Idx3d) -> bool {
        let (z, h, w) = self.shape();
        *z0 < z && *h0 < h && *w0 < w
    }

    /// 获取该 3D CT 文件的方向信息.
    ///
    /// # 注意
    ///
    /// 该算法在 LiTS 数据集上保证正确, 但可能不适用于普通 3D CT 腹部扫描.
    /// 后面可能会扩展以解决此问题. fixme.
    #[inline]
    fn lls_sector_pattern(&self) -> Result<LlsSectorPattern, InitLlsPatternError> {
        LlsSectorPattern::from_header(self.header())
    }

    /// 获取单个体素分辨率. 该分辨率以毫米为单位, 分别代表空间 (相邻切片方向),
    /// 高 (自然图像的垂直方向), 宽 (自然图像的水平方向).
    ///
    /// 该值也可以通过 `self.{z_mm, height_mm, width_mm}` 分别获取.
    #[inline]
    fn pix_dim(&self) -> [f64; 3] {
        let [_, w, h, z, ..] = self.header().pixdim;
        [z as f64, h as f64, w as f64]
    }

    /// 获取 width 方向 (自然 2D 图像的水平方向) 体素分辨率, 以毫米为单位.
    #[inline]
    fn width_mm(&self) -> f64 {
        self.header().pixdim[1] as f64
    }

    /// 获取 height 方向 (自然 2D 图像的垂直方向) 体素分辨率, 以毫米为单位.
    #[inline]
    fn height_mm(&self) -> f64 {
        self.header().pixdim[2] as f64
    }

    /// 获取空间方向 (相邻 2D 切片的方向) 体素分辨率, 以毫米为单位.
    #[inline]
    fn z_mm(&self) -> f64 {
        self.header().pixdim[3] as f64
    }

    /// 体素在侧视图上看是否是 "矮胖" 的?
    #[inline]
    fn is_height_greater(&self) -> bool {
        self.height_mm() > self.z_mm()
    }

    /// 体素在侧视图上看是否是 "瘦高" 的?
    #[inline]
    fn is_z_greater(&self) -> bool {
        self.z_mm() > self.height_mm()
    }

    /// 体素分辨率在三个维度上是否是各向同的?
    #[inline]
    fn is_isotropic(&self) -> bool {
        let [z, h, w] = self.pix_dim();
        z == h && z == w
    }

    /// 获取体素的实际体积值, 以立方毫米为单位.
    #[inline]
    fn voxel(&self) -> f64 {
        self.pix_dim().iter().product()
    }

    /// 获取水平切片方向的像素实际面积值, 以平方毫米为单位.
    #[inline]
    fn slice_pixel(&self) -> f64 {
        self.pix_dim().iter().skip(1).product()
    }
}

impl NiftiHeaderAttr for CtScan {
    #[inline]
    fn header(&self) -> &NiftiHeader {
        &self.header
    }
}

impl Index<Idx3d> for CtScan {
    type Output = f32;

    #[inline]
    fn index(&self, index: Idx3d) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<Idx3d> for CtScan {
    #[inline]
    fn index_mut(&mut self, index: Idx3d) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl CtScan {
    /// 打开 nii 文件格式的 3D CT 扫描. `path` 为 nii 文件的本地路径.
    /// 如果打开成功, 则返回 `Ok(Self)`, 否则返回 `Err`.
    pub fn open<P: AsRef<Path>>(path: P) -> nifti::Result<Self> {
        let obj = ReaderOptions::new().read_file(path.as_ref())?;
        let header = Box::new(obj.header().clone());

        // [W, H, z] -> [z, H, W].
        // hint: 原第一维向下增长, 原第二维向右增长.
        let data = obj
            .into_volume()
            .into_ndarray()?
            .permuted_axes([2, 1, 0].as_slice());

        // The nature of nifti data field layout.
        debug_assert!(data.is_standard_layout());

        // 该操作不会生成 `Err`, 可直接 unwrap.
        let data =
            Array3::<f32>::from_shape_vec(get_shape_from_header(&header), data.into_raw_vec())
                .unwrap();

        Ok(Self { header, data })
    }

    /// 计算由 `it` 给出的所有索引对应的 CT HU 值的平均值.
    ///
    /// 如果存在越界索引, 则程序 panic.
    pub fn mean_hu<I: IntoIterator<Item = Idx3d>>(&self, it: I) -> f64 {
        let mut count = 0u64;
        let mut hu = 0.0;
        for pos in it.into_iter() {
            count += 1;
            hu += self[pos] as f64;
        }
        hu / (count as f64)
    }

    /// 计算第 `z_index` 个水平切片上的、由 `it` 给出的所有索引对应的 CT HU 值的平均值.
    ///
    /// 如果 `z_index` 或 `it` 中的二维索引越界, 则程序 panic.
    pub fn mean_hu_2d<I: IntoIterator<Item = Idx2d>>(&self, it: I, z_index: usize) -> f64 {
        let sli = self.slice_at(z_index);
        let mut count = 0u64;
        let mut hu = 0.0;
        for pos in it.into_iter() {
            count += 1;
            hu += sli[pos] as f64;
        }
        hu / (count as f64)
    }

    /// 获取 3D 扫描 z 空间的第 `z_index` 层切片视图.
    ///
    /// 当 `z_index` 越界时 panic.
    #[inline]
    pub fn slice_at(&self, z_index: usize) -> ScanSlice<'_> {
        ScanSlice::new(self.data.index_axis(Axis(0), z_index))
    }

    /// 获取 3D 扫描 z 空间的第 `z_index` 层可变切片视图.
    ///
    /// 当 `z_index` 越界时 panic.
    #[inline]
    pub fn slice_at_mut(&mut self, z_index: usize) -> ScanSliceMut<'_> {
        ScanSliceMut::new(self.data.index_axis_mut(Axis(0), z_index))
    }

    /// 获取能按升序迭代 3D 扫描水平不可变切片的迭代器.
    #[inline]
    pub fn slice_iter(&self) -> impl ExactSizeIterator<Item = ScanSlice> {
        self.data.axis_iter(Axis(0)).map(ScanSlice::new)
    }

    /// 获取能按升序迭代 3D 扫描水平可变切片的迭代器.
    #[inline]
    pub fn slice_iter_mut(&mut self) -> impl ExactSizeIterator<Item = ScanSliceMut> {
        self.data.axis_iter_mut(Axis(0)).map(ScanSliceMut::new)
    }

    /// 获得数据的一份不可变 shallow copy.
    #[inline]
    pub fn data(&self) -> ArrayView<'_, f32, Ix3> {
        self.data.view()
    }

    /// 获得数据的一份可变 shallow copy.
    #[inline]
    pub fn data_mut(&mut self) -> ArrayViewMut<'_, f32, Ix3> {
        self.data.view_mut()
    }
}

/// nii 格式 3D CT 标注, 包括 header 和真值标签. 标签值以 `u8` 保存.
#[derive(Debug, Clone)]
pub struct CtLabel {
    header: BoxedHeader,
    data: Array3<u8>,
}

impl NiftiHeaderAttr for CtLabel {
    #[inline]
    fn header(&self) -> &NiftiHeader {
        &self.header
    }
}

impl Index<Idx3d> for CtLabel {
    type Output = u8;

    #[inline]
    fn index(&self, index: Idx3d) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<Idx3d> for CtLabel {
    #[inline]
    fn index_mut(&mut self, index: Idx3d) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl CtLabel {
    /// 打开 nii 文件格式的 3D CT 标注. `path` 为 nii 文件的本地路径. 如果打开成功,
    /// 则返回 `Ok(Self)`, 否则返回 `Err`.
    pub fn open<P: AsRef<Path>>(path: P) -> nifti::Result<Self> {
        let obj = ReaderOptions::new().read_file(path.as_ref())?;
        let header = Box::new(obj.header().clone());

        // [W, H, z] -> [z, H, W]
        // hint: 原第一维向下增长, 原第二维向右增长.
        let data = obj
            .into_volume()
            .into_ndarray::<u8>()?
            .permuted_axes([2, 1, 0].as_slice());

        // The nature of nifti data field layout.
        debug_assert!(data.is_standard_layout());

        // 该操作不会生成 `Err`, 可直接 unwrap.
        let data =
            Array3::<u8>::from_shape_vec(get_shape_from_header(&header), data.into_raw_vec())
                .unwrap();

        Ok(Self { header, data })
    }

    /// 根据裸标签数据和部分元信息直接创建 `CtLabel` 实体.
    /// 目前, 如果参数数据不符合规定则返回 `None`.
    ///
    /// # 参数
    ///
    /// 1. `data` 的数据必须非空, 且为 0, 1 或 2. 否则程序行为未定义.
    /// 2. `data` 按照 nifti 惯用标准以 \[w, h, z\] 格式存储.
    /// 3. `pix_dim` 按照 \[w, h, z\] 格式存储.
    ///
    /// # 注意
    ///
    /// 该方法可能会创建不一致的实体, 因此你应仅将其用于实验目的.
    pub fn fake(
        data: Array3<u8>,
        pix_dim: [f32; 3],
        qform_code: i16,
        quatern_bcd: [f32; 3],
    ) -> Self {
        let data = data.permuted_axes([2, 1, 0]);
        let data = if data.is_standard_layout() {
            data
        } else {
            data.as_standard_layout().to_owned()
        };
        debug_assert!(data.is_standard_layout());

        let mut header = Box::<NiftiHeader>::default();
        let [_, pw, ph, pz, ..] = &mut header.pixdim;
        let [w, h, z] = &pix_dim;
        assert_eq!(w, h); // 目前仅支持水平方向各向同性的情况
        (*pw, *ph, *pz) = (*w, *h, *z);
        header.qform_code = qform_code;
        let [qb, qc, qd] = &quatern_bcd;
        (header.quatern_b, header.quatern_c, header.quatern_d) = (*qb, *qc, *qd);
        // header.intent_name.take(4).
        header.intent_name[..4].copy_from_slice(b"fake");

        Self { header, data }
    }

    /// 直接创建数据.
    ///
    /// # 注意
    ///
    /// **目前** 你应当使输入满足以下性质, 否则程序行为未定义:
    ///
    /// 1. `data` 按照 \[width, height, z\] 组织, 内部体素值非空,
    ///   且必须为 0, 1 或 2.
    /// 2. `header` 必须满足其格式标准, 目前仅支持从 LiTS
    ///   训练集中已有的 label header 加载.
    #[inline]
    pub fn fake_with_header(header: &NiftiHeader, data: Array3<u8>) -> Self {
        let data = data.permuted_axes([2, 1, 0]);
        let data = if data.is_standard_layout() {
            data
        } else {
            data.as_standard_layout().to_owned()
        };
        debug_assert!(data.is_standard_layout());

        let mut header = Box::new(header.clone());
        header.intent_name[..4].copy_from_slice(b"fake");
        Self { header, data }
    }

    /// 判断该结构是否是由 `fake_*` 方法手动拼接的.
    pub fn is_faked(&self) -> bool {
        self.header.intent_name.starts_with(b"fake")
    }

    /// 获取 3D 标注 z 空间的第 `z_index` 层不可变切片.
    ///
    /// 当 `z_index` 越界时 panic.
    #[inline]
    pub fn slice_at(&self, z_index: usize) -> LabelSlice {
        LabelSlice::new(self.data.index_axis(Axis(0), z_index))
    }

    /// 获取 3D 标注 z 空间的第 `z_index` 层可变切片.
    ///
    /// 当 `z_index` 越界时 panic.
    #[inline]
    pub fn slice_at_mut(&mut self, z_index: usize) -> LabelSliceMut {
        LabelSliceMut::new(self.data.index_axis_mut(Axis(0), z_index))
    }

    /// 获取能按升序迭代 3D 标注水平不可变切片的迭代器.
    #[inline]
    pub fn slice_iter(&self) -> impl ExactSizeIterator<Item = LabelSlice> {
        self.data.axis_iter(Axis(0)).map(LabelSlice::new)
    }

    /// 获取能按升序迭代 3D 标注水平可变切片的迭代器.
    #[inline]
    pub fn slice_iter_mut(&mut self) -> impl ExactSizeIterator<Item = LabelSliceMut> {
        self.data.axis_iter_mut(Axis(0)).map(LabelSliceMut::new)
    }

    /// 获得数据的一份不可变 shallow copy.
    #[inline]
    pub fn data(&self) -> ArrayView<'_, u8, Ix3> {
        self.data.view()
    }

    /// 获得数据的一份可变 shallow copy.
    #[inline]
    pub fn data_mut(&mut self) -> ArrayViewMut<'_, u8, Ix3> {
        self.data.view_mut()
    }

    /// 获取 3D 标注中值为 `label` 的体素个数.
    #[inline]
    pub fn count(&self, label: u8) -> usize {
        self.data.iter().filter(|p| **p == label).count()
    }

    /// 获取 CT 标签的基本统计信息.
    ///
    /// 统计信息格式为: \[背景像素数, 肝脏像素数, 肿瘤像素数\].
    /// 该操作不会统计任何其他像素信息.
    pub fn numeric_statistics(&self) -> [usize; 3] {
        let mut ans = [0; 3];
        for pixel in self.data.iter().filter(|p| **p <= 2) {
            ans[*pixel as usize] += 1;
        }
        ans
    }

    /// 将 3D 标注中值为 `old` 的体素全部替换为 `new`.
    ///
    /// 返回总共成功替换的个数.
    pub fn replace(&mut self, old: u8, new: u8) -> usize {
        let mut cnt = 0usize;
        self.data_mut()
            .iter_mut()
            .filter(|pix| **pix == old)
            .for_each(|p| {
                cnt += 1;
                *p = new;
            });
        cnt
    }

    /// 收集满足谓词 `pred` 的所有像素对应的下标, 结果按行优先存储.
    pub fn filter_pos(&self, pred: Predicate) -> Vec<Idx3d> {
        self.data
            .indexed_iter()
            .filter_map(|(ref pos, pixel)| pred(*pixel).then_some(*pos))
            .collect()
    }

    /// 收集所有肝脏 + 肿瘤像素对应的下标. 结果按行优先存储.
    #[inline]
    pub fn liver_tumor_pos(&self) -> Vec<Idx3d> {
        self.filter_pos(|p| matches!(p, LITS_LIVER | LITS_TUMOR))
    }

    /// 收集所有肝脏像素对应的下标. 结果按行优先存储.
    #[inline]
    pub fn liver_pos(&self) -> Vec<Idx3d> {
        self.filter_pos(|p| matches!(p, LITS_LIVER))
    }

    /// 收集所有肿瘤像素对应的下标. 结果按行优先存储.
    #[inline]
    pub fn tumor_pos(&self) -> Vec<Idx3d> {
        self.filter_pos(|p| matches!(p, LITS_TUMOR))
    }

    /// 将三维标签中的背景空洞
    /// (即钻石-连通背景区域中面积不是最大的那些)
    /// 填充为肝脏像素.
    ///
    /// # 鲁棒性
    ///
    /// 该算法鲁棒性的一个充分条件是三维标签的边缘像素
    /// (长方体的六个表面所代表的像素的并集) 全为背景.
    /// 因此在算法运行前 **会强制将六个表面设置为背景**.
    pub fn fill_background_hollow(&mut self) -> bool {
        self.make_background_surface6();
        let mut non_trivial = false;
        self.slice_iter_mut()
            .for_each(|mut s| non_trivial |= s.fill_background_hollow());
        // self.par_for_each_slice_mut(|mut s| {s.fill_background_hollow(); });
        non_trivial
    }

    /// 将 3D 图像的六个表面矩形填充为背景.
    fn make_background_surface6(&mut self) {
        debug_assert_ne!(self.size(), 0);
        let (z, h, w) = self.shape();
        macro_rules! fill_bg {
            ($axis: expr, $index: expr) => {
                self.data
                    .index_axis_mut(Axis($axis), $index)
                    .fill(LITS_BACKGROUND);
                debug_assert!(
                    LabelSlice::new(self.data.index_axis(Axis($axis), $index)).is_background()
                );
            };
        }
        fill_bg!(0, 0);
        fill_bg!(0, z - 1);
        fill_bg!(1, 0);
        fill_bg!(1, h - 1);
        fill_bg!(2, 0);
        fill_bg!(2, w - 1);
    }

    /// 获取 `pos` 前后上下左右六个点的坐标.
    ///
    /// 在数据范围外的坐标会被过滤掉, 不会包含在返回值中.
    fn diamond_neighbours(&self, (z, h, w): Idx3d) -> Vec<Idx3d> {
        self.check_collect([
            (z.wrapping_sub(1), h, w),
            (z.saturating_add(1), h, w),
            (z, h.wrapping_sub(1), w),
            (z, h.saturating_add(1), w),
            (z, h, w.wrapping_sub(1)),
            (z, h, w.saturating_add(1)),
        ])
    }

    /// 获取 `pos` 上下两个点的坐标.
    ///
    /// 在数据范围外的坐标会被过滤掉, 不会包含在返回值中.
    fn neighbours_z(&self, (z, h, w): Idx3d) -> Vec<Idx3d> {
        self.check_collect([(z.wrapping_sub(1), h, w), (z.saturating_add(1), h, w)])
    }

    /// 获取 `pos` 上下左右四个点的坐标.
    ///
    /// 在数据范围外的坐标会被过滤掉, 不会包含在返回值中.
    fn neighbours_hw(&self, (z, h, w): Idx3d) -> Vec<Idx3d> {
        self.check_collect([
            (z, h.wrapping_sub(1), w),
            (z, h.saturating_add(1), w),
            (z, h, w.wrapping_sub(1)),
            (z, h, w.saturating_add(1)),
        ])
    }

    /// 收集 `dataset` 中不越界的索引.
    #[inline]
    fn check_collect<B: FromIterator<Idx3d>, const N: usize>(&self, data: [Idx3d; N]) -> B {
        data.into_iter().filter(|p| self.check(p)).collect()
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "rayon")] {
        use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    }
}

/// 并发操作部分
#[cfg(feature = "rayon")]
impl CtLabel {
    /// 借助 `rayon`, 并行地对 3D 标注每个水平可变切片实施 `op` 操作.
    pub fn par_for_each_slice_mut<F>(&mut self, op: F)
    where
        F: Fn(LabelSliceMut) + Sync + Send,
    {
        self.data_mut()
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|v| {
                op(LabelSliceMut::new(v));
            });
    }

    /// 借助 `rayon`, 并行地对 3D 标注每个水平不可变切片实施 `op` 操作.
    pub fn par_for_each_slice<F>(&self, op: F)
    where
        F: Fn(LabelSlice) + Sync + Send,
    {
        self.data()
            .axis_iter(Axis(0))
            .into_par_iter()
            .for_each(|v| {
                op(LabelSlice::new(v));
            });
    }

    /// 借助 `rayon`, 并行地对 3D 标注每个水平可变切片实施 `op` 操作.
    /// 该操作会同时携带 z 方向索引信息.
    pub fn par_for_each_indexed_slice_mut<F>(&mut self, op: F)
    where
        F: Fn(usize, LabelSliceMut) + Sync + Send,
    {
        self.data_mut()
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, v)| {
                op(i, LabelSliceMut::new(v));
            });
    }

    /// 借助 `rayon`, 并行地对 3D 标注每个水平不可变切片实施 `op` 操作.
    /// 该操作会同时携带 z 方向索引信息.
    pub fn par_for_each_indexed_slice<F>(&self, op: F)
    where
        F: Fn(usize, LabelSlice) + Sync + Send,
    {
        self.data()
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, v)| {
                op(i, LabelSlice::new(v));
            });
    }

    /// 借助 `rayon`, 并行地将 3D 标注中值为 `old` 的体素全部替换为 `new`.
    ///
    /// 返回总共成功替换的个数.
    pub fn par_replace(&mut self, old: u8, new: u8) -> usize {
        let cnt = AtomicUsize::new(0);
        self.data_mut()
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|v| {
                let mut sli = LabelSliceMut::new(v);
                let local = sli.replace(old, new);
                cnt.fetch_add(local, Ordering::Release);
            });

        cnt.load(Ordering::Acquire)
    }

    /// 借助 `rayon`, 使用多任务模式运行 `self.fill_background_hollow`.
    pub fn par_fill_background_hollow(&mut self) -> bool {
        self.make_background_surface6();
        let non_trivial = AtomicBool::new(false);
        self.par_for_each_slice_mut(|mut s| {
            non_trivial.fetch_or(s.fill_background_hollow(), Ordering::Release);
        });
        non_trivial.load(Ordering::Acquire)
    }
}

/// nii 格式的 3D CT 扫描与对应的标注.
///
/// 该结构完全透明, 仅包含两个公开的 `scan` 和 `label` 子结构,
/// 用户可以直接使用它们来实现相关上层功能.
///
/// # 注意
///
/// 两个子结构的数据一致性由用户保证, 否则程序行为未定义.
#[derive(Debug, Clone)]
pub struct CtData3d {
    /// 3D CT 扫描.
    pub scan: CtScan,

    /// 3D CT 标注.
    pub label: CtLabel,
}

impl CtData3d {
    /// 分别打开 nii 文件格式的 3D CT 扫描和对应标注. 如果任一文件打开失败, 则返回 `Err`.
    /// 若两个文件的数据文件形状不一致, 则程序 `panic`.
    pub fn open(scan_path: impl AsRef<Path>, label_path: impl AsRef<Path>) -> nifti::Result<Self> {
        let scan = CtScan::open(scan_path.as_ref())?;
        let label = CtLabel::open(label_path.as_ref())?;
        assert_eq!(scan.shape(), label.shape(), "CT 扫描和标注形状不一致");
        Ok(Self { scan, label })
    }

    /// 获取水平切片个数.
    #[inline]
    pub fn len_z(&self) -> usize {
        self.label.len_z()
    }

    /// 依次获取 3D 标注和 3D 扫描 z 空间的第 `z_index` 层不可变切片.
    ///
    /// 当 `z_index` 越界时 panic.
    #[inline]
    pub fn slice_at(&self, z_index: usize) -> (ScanSlice<'_>, LabelSlice<'_>) {
        (self.scan.slice_at(z_index), self.label.slice_at(z_index))
    }

    /// 依次获取 3D 标注和 3D 扫描 z 空间的第 `z_index` 层可变切片.
    ///
    /// 当 `z_index` 越界时 panic.
    #[inline]
    pub fn slice_at_mut(&mut self, z_index: usize) -> (ScanSliceMut<'_>, LabelSliceMut<'_>) {
        (
            self.scan.slice_at_mut(z_index),
            self.label.slice_at_mut(z_index),
        )
    }

    /// 获取能按升序迭代 3D 水平 (扫描, 标注) 不可变切片的迭代器.
    #[inline]
    pub fn slice_iter(&self) -> impl ExactSizeIterator<Item = (ScanSlice, LabelSlice)> {
        self.scan.slice_iter().zip(self.label.slice_iter())
    }

    /// 获取能按升序迭代 3D 水平 (扫描, 标注) 可变切片的迭代器.
    pub fn slice_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (ScanSliceMut, LabelSliceMut)> {
        self.scan.slice_iter_mut().zip(self.label.slice_iter_mut())
    }

    /// 获取能按行优先序迭代 3D (扫描, 标注) 像素的迭代器.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&f32, &u8)> {
        self.scan.data.iter().zip(self.label.data.iter())
    }
}
