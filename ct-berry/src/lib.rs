#![warn(missing_docs)] // <= 合适时移除它.
// #![warn(clippy::missing_docs_in_private_items)]  // <= too strict.

//! 核心库. 提供 LiTS 数据集和医院数据集的肝脏 (及肿瘤) 文件的结构化信息和基础处理算法.
//!
//! 该 crate 目前仅提供 `safe` 接口. 将来可能为部分高性能场景关键路径提供 `unsafe` 接口.
//!
//! # 注意
//!
//! 1. 该 crate 目前主要负责处理 LiTS 数据, 没有对其它源的数据进行直接适配
//!   (但如果新数据按照 LiTS 模式进行组织, 也可以工作).
//! 2. 在非期望情况下, 程序会直接 panic, 而不会导致内存错误. As what Rust promises.
//!
//! # 开发计划
//!
//! ### 8-邻域图像内容提取与相关消融实验 ✅
//!
//! 实现位于 `ct-berry/src/eight`.
//!
//! ### 三次样条插值 & 最小二乘多项式拟合的纯 Rust 实现 ✅
//!
//! 曲线拟合功能.
//!
//! 实现位于 `ct-berry/src/fitting`.
//!
//! ### 三维形态学操作 ✅
//!
//! center ROI 提取和 peripheral ROI 提取两部分.
//!
//! 实现位于 `ct-berry/src/dataset/morph_3d`.
//!
//! ### 肝衰减窗口实现 ✅
//!
//! 提供了计算给定区域 CT HU 平均值的功能.
//!
//! 参考论文: "Fully automatic liver attenuation estimation
//! combing CNN segmentation and morphological operations".
//! 从该论文得知计算肝衰减的近似方法.
//!
//! ### CT window 视图 ✅
//!
//! 提供一个独立的 CT 窗口对象, 以便将 CT HU 值转换为 8-bit 灰度值.
//!
//! 实现位于 `ct-berry/src/data/window.rs`.
//!
//! ### 根据肝衰减值来精化肝脏标签轮廓, 及其 benchmark 框架 ⌛️
//!
//! 从语义分割结果 (或直接真值标签) 出发, 通过精化获取进一步结果.
//!
//! 算法效果量化 ⌛️
//!
//! 实现位于 `ct-berry/src/post_proc/surface_refine`.
//!
//! ### 判断一个像素是否在一个扇形区域内 ✅
//!
//! 评估区域像素内的 TP/TN/FP/FN, 评估肝衰减选取的有效性.
//!
//! 实现位于 `ct-berry/src/data/sector`.
//!
//! ### 从 CT 切片定位左前叶稀疏像素链 ✅
//!
//! 水平切片形态学腐蚀到中心, 然后发出两条射线. 根据此 `Sector`
//! 提取实际肝左外区.
//!
//! 实现位于 `ct-berry/src/post_proc/locate_lls.rs`.
//!
//! ### 拟合点采样, 像素坐标系偏移, LSN 计算 ✅
//!
//! 1. 如何处理像素/体素和图像质点之间的差异? ✅
//! 2. 如何采样肝脏表面像素 (决定以多大的实际距离为单位采一次点)? ✅
//! 3. 如何采样拟合的曲线点? ✅
//! 4. 如何计算上述两个要素之间的最短线段距离? ✅
//! 5. 如何将图像坐标系转换为自然坐标系? ✅
//!
//! 上述问题的答案:
//!
//! 1. 按照原始像素索引出发, 拟合出的曲线进行 `+(0.5, +0.5)` 修正即可.
//!   如果需要画图, 则对每个像素进行 `C * C` 扩展, 并以 `C` 为倍率扩展修正拟合点.
//! 2. 可以 (1) 按照固定距离采样, 或 (2) 按照固定点均距采样.
//!   为了让曲线的弯曲程度在不同图像 (尤其是像素自身分辨率不同时) 保持一致性,
//!   我们倾向于选 (1). 程序同时提供了这两种方式便于对比.
//! 3. 以 "点数每像素宽" 为单位, 默认可以选择 10.
//! 4. 计算每个肝脏表面像素到拟合曲线的距离, 并求平均值.
//! 5. lsn 模块接受正常的 `(h, w)` 格式输入, 将其内部转换成 `(x, y)`,
//!   处理后再转换回去. 所谓 `(x, y)` 即正常平面直角坐标系.
//!   `(in_h, in_w)` -> `(in_w, H - in_h - 1)` == `(x, y)`
//!   -> `(H - y - 1, x)` = `(out_h, out_w)`
//!
//! 实现位于 `ct-berry/src/lsn/*`.
//!
//! ### 小功能 ✅
//!
//! 1. 提供 mirror type 以支持 CT 切片的备份与恢复. ✅
//! 2. 扩展 `struct Profile` 功能以提供更有价值的统计学信息. ✅
//! 3. Data iterator ✅
//!
//! ### 完善代码文档 ✅
//!
//! 给每个 public API 提供文档, 并视情况给 private
//! API 提供文档.

/// 二维索引, 同时也可一定程度上用作非负整数向量.
pub type Idx2d = (usize, usize);

/// 三维索引, 同时也可一定程度上用作非负整数向量.
pub type Idx3d = (usize, usize, usize);

/// 高精度通用索引 / 向量.
type Idx2dF = (f64, f64);

/// 压缩存储优化时会用到. 该结构不对外公开.
type Idx2dU16 = (u16, u16);

/// 压缩存储优化时会用到. 该结构不对外公开.
type Idx3dU16 = (u16, u16, u16);
type Predicate = fn(u8) -> bool;

type Area2d = Vec<Idx2d>;
type Areas2d = Vec<Area2d>;

/// 3D CT nii 文件基础数据结构.
mod data;

pub use data::{
    CompactLabelSlice, CtData3d, CtLabel, CtScan, CtWindow, ImgWriteRaw, ImgWriteVis, LabelSlice,
    LabelSliceMut, NiftiHeaderAttr, OwnedLabelSlice, OwnedScanSlice, ScanSlice, ScanSliceMut,
};

pub use data::sector;

pub mod consts;

#[cfg(feature = "plot")]
pub use data::ImgDisplay;

pub mod eight;

pub mod fitting;

pub mod post_proc;

pub mod dataset;
pub mod lsn;
pub mod prelude;
