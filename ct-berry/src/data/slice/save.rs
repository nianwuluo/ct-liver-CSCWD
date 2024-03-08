//! 图像的持久化存储.

use crate::{LabelSlice, LabelSliceMut, ScanSlice, ScanSliceMut};
use image::ImageResult;
use std::path::Path;

/// 表明一个可以通过 **可视化友好** 模式持久化存储的图像对象.
///
/// `ImgWriteVis` trait 的意图是, 图像将以 "可视化友好"
/// 的方式保存, 而不是 "as is" 的方式. 这意味着, 对于 `LabelSlice`, `LabelSliceMut`
/// 这类仅存在 0, 1, 2, 3 像素值的图像, 在保存时会映射到肉眼较易能区分的形式;
/// 对于 `ScanSlice`, `ScanSliceMut` 这类以 CT HU 值存储的扫描,
/// 在保存时会用常见的肝脏可视化窗口规范化.
pub trait ImgWriteVis {
    /// 按照一定的可视化规则将图片保存到 `path` 路径.
    fn save<P: AsRef<Path>>(&self, path: P) -> ImageResult<()>;
}

/// 表明一个可以通过 **按原样** 模式持久化存储的图像对象.
///
/// `ImgWriteRaw` trait 的额外意图是, 图像将按原样保存. 这意味着,
/// 对于 `LabelSlice`, `LabelSliceMut` 这类图像可以直接存储会都图像,
/// 但面对 `ScanSlice`, `ScanSliceMut` 这类以 CT HU 值存储的扫描无能为力.
pub trait ImgWriteRaw {
    /// 按原样将图片保存到 `path` 路径.
    fn save_raw<P: AsRef<Path>>(&self, path: P) -> ImageResult<()>;
}

/// 使像素更有利于单通道可视化.use
#[inline]
pub(crate) fn pretty(label: u8) -> u8 {
    use crate::consts::gray::*;
    match label {
        // 背景为黑色
        LITS_BACKGROUND => BLACK,

        // 肝脏为白色
        LITS_LIVER => WHITE,

        // 让肿瘤颜色更接近肝脏颜色
        LITS_TUMOR => LIGHT_GRAY,

        // 让轮廓颜色更接近背景颜色
        LITS_BOUNDARY => DARK_GRAY,

        any_else => panic!("只允许图像存在 0, 1, 2, 3 像素, 但发现了 `{any_else}`"),
    }
}

macro_rules! impl_label_vis {
    ($($slice: ty),+) => {
        $(
            /// 会将背景/肝脏/肿瘤/表面像素分别映射为黑色/白色/亮灰色/暗灰色. 不允许其他颜色.
            impl ImgWriteVis for $slice {
                fn save<P: AsRef<Path>>(&self, path: P) -> image::ImageResult<()> {
                    let (height, width) = self.shape();
                    let mut buf = image::GrayImage::new(width as u32, height as u32);
                    for ((h, w), &pix) in self.indexed_iter() {
                        buf.put_pixel(w as u32, h as u32, image::Luma([pretty(pix)]));
                    }
                    buf.save(path)
                }
            }
        )+
    };
}

macro_rules! impl_label_raw {
    ($($slice: ty),+) => {
        $(
            /// 按原样存储.
            impl ImgWriteRaw for $slice {
                fn save_raw<P: AsRef<Path>>(&self, path: P) -> image::ImageResult<()> {
                    let (height, width) = self.shape();
                    let mut buf = image::GrayImage::new(width as u32, height as u32);
                    for ((h, w), &pix) in self.indexed_iter() {
                        buf.put_pixel(w as u32, h as u32, image::Luma([pix]));
                    }
                    buf.save(path)
                }
            }
        )+
    };
}

macro_rules! impl_scan_vis {
    ($($scan: ty),+) => {
        $(
            /// 窗位 60, 窗宽 200.
            impl ImgWriteVis for $scan {
                fn save<P: AsRef<Path>>(&self, path: P) -> ImageResult<()> {
                    let (height, width) = self.shape();
                    let mut buf = image::GrayImage::new(width as u32, height as u32);
                    const WINDOW: crate::CtWindow = crate::CtWindow::from_liver_visual();
                    for ((h, w), &hu) in self.indexed_iter() {
                        let gray = WINDOW.eval(hu).unwrap();
                        buf.put_pixel(w as u32, h as u32, image::Luma([gray]));
                    }
                    buf.save(path)
                }
            }
        )+
    };
}

impl_label_vis!(LabelSlice<'_>, LabelSliceMut<'_>);
impl_scan_vis!(ScanSlice<'_>, ScanSliceMut<'_>);
impl_label_raw!(LabelSlice<'_>, LabelSliceMut<'_>);
