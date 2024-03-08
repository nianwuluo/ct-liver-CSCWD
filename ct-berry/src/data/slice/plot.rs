//! 图片展示和保存模块, 主要用于调试.
//!
//! # 注意
//!
//! 需要 `plot` feature.

use crate::{Idx2d, LabelSlice, LabelSliceMut, ScanSlice, ScanSliceMut};
use ndarray::{ArcArray2, Array2, ArrayView2, CowArray, Ix2};
use opencv::highgui::{imshow, wait_key};
use opencv::prelude::{Mat, MatTrait, MatTraitConst};
use std::time::Duration;

/// 表明一个可以在窗口中可视化的对象.
pub trait ImgDisplay {
    /// 展示对象.
    fn show(&self);

    /// 同 `show()`, 但在之后自动等待一次用户按键输入.
    fn show_and_wait(&self) {
        self.show();
        wait_key(0).unwrap(); // never fails
    }

    /// 同 `show()`, 但在之后自动等待给定时间.
    fn show_and_wait_for(&self, d: Duration) -> opencv::Result<i32> {
        self.show();
        let ms = d.as_millis();
        assert!((i32::MAX as u128) < ms);
        wait_key(ms as i32)
    }
}

/// 将 `dataset` 按行优先格式, 以 `shape` 分辨率存储为矩阵.
/// 会额外进行可视化友好的像素转换.
fn label_slice_to_opencv_mat(data: &[u8], (h, w): Idx2d) -> Mat {
    assert_eq!(data.len(), h * w);
    let mut mat = Mat::from_slice_rows_cols(data, h, w).unwrap();

    let size = mat.size().unwrap();
    debug_assert_eq!(size.height as usize, h);
    debug_assert_eq!(size.width as usize, w);

    for i in 0..size.height {
        for j in 0..size.width {
            let slot = mat.at_2d_mut::<u8>(i, j).unwrap();
            *slot = super::save::pretty(*slot);
        }
    }
    mat
}

/// 将 `dataset` 按行优先格式, 以 `shape` 分辨率存储为矩阵.
/// 会额外进行可视化友好的像素转换 (窗位 60, 窗宽 200).
fn scan_slice_to_opencv_mat(data: ArrayView2<f32>, (h, w): Idx2d) -> Mat {
    use opencv::core::{Scalar, Size, CV_8UC1};

    assert_eq!(data.len(), h * w);
    let mut mat =
        Mat::new_size_with_default(Size::new(w as i32, h as i32), CV_8UC1, Scalar::from(0))
            .unwrap();

    let size = mat.size().unwrap();
    debug_assert_eq!(size.height as usize, h);
    debug_assert_eq!(size.width as usize, w);

    const WINDOW: crate::CtWindow = crate::CtWindow::from_liver_visual();
    for i in 0..size.height {
        for j in 0..size.width {
            let slot = mat.at_2d_mut::<u8>(i, j).unwrap();
            *slot = WINDOW.eval(data[(i as usize, j as usize)]).unwrap();
        }
    }
    mat
}

macro_rules! label_slice_outer_doc {
    () => { "该对象最多只允许 `0`, `1`, `2`, `3` 值, 分别代表 `LITS_BACKGROUND`, `LITS_LIVER`, `LITS_TUMOR`, `LITS_BOUNDARY`." };
}
macro_rules! label_slice_show_doc {
    () => {
        r"为了获得更清晰的可视化对象, 该功能在展示前对颜色像素值做如下映射:

0 (背景, `LITS_BACKGROUND`) -> 0 (黑色);

1 (肝脏, `LITS_LIVER`) -> 255 (白色);

2 (肿瘤, `LITS_TUMOR`) -> 192 (亮灰色);

3 (边缘, `LITS_BOUNDARY`) -> 64 (暗灰色)."
    };
}

macro_rules! impl_label {
    (Slice, {$($slice: ty),+}) => {
        $(
            #[doc = label_slice_outer_doc!()]
            impl ImgDisplay for $slice {
                #[doc = label_slice_show_doc!()]
                fn show(&self) {
                    let mat = label_slice_to_opencv_mat(self.data().as_slice().unwrap(), self.shape());
                    imshow("Image", &mat).unwrap();
                }
            }
        )+
    };
    (Array, {$($array: ty),+}) => {
        $(
            #[doc = label_slice_outer_doc!()]
            impl ImgDisplay for $array {
                #[doc = label_slice_show_doc!()]
                fn show(&self) {
                    let &[h, w] = self.shape() else { unreachable!() };
                    let mat = if let Some(sli) = self.as_slice() {
                        label_slice_to_opencv_mat(sli, (h, w))
                    } else {
                        let sl = self.as_standard_layout().clone();
                        label_slice_to_opencv_mat(sl.as_slice().unwrap(), (h, w))
                    };
                    imshow("Image", &mat).unwrap();
                }
            }
        )+
    };
}

macro_rules! impl_scan {
    ({$($scan: ty),+}) => {
        $(
            /// 可视化扫描.
            impl ImgDisplay for $scan {
                fn show(&self) {
                    let mat = scan_slice_to_opencv_mat(self.data().view(), self.shape());
                    imshow("Image", &mat).unwrap();
                }
            }
        )+
    };
}

impl_label!(Slice, {LabelSlice<'_>, LabelSliceMut<'_>});
impl_label!(Array, {Array2<u8>, CowArray<'_, u8, Ix2>, ArrayView2<'_, u8>, ArcArray2<u8>});
impl_scan!({ScanSlice<'_>, ScanSliceMut<'_>});
