//! 通用常量.

/// 单通道颜色.
pub mod gray {
    /// 原 LiTS 数据集中, 背景的像素值.
    pub const LITS_BACKGROUND: u8 = 0;

    /// 原 LiTS 数据集中, 肝脏的像素值.
    pub const LITS_LIVER: u8 = 1;

    /// 原 LiTS 数据集中, 肿瘤的像素值.
    pub const LITS_TUMOR: u8 = 2;

    /// LiTS 数据集切片中, 肝脏边缘的 (预留) 像素值.
    pub const LITS_BOUNDARY: u8 = 3;

    /// 单通道黑色.
    pub const BLACK: u8 = 0b_0000_0000;

    /// 单通道暗灰色.
    pub const DARK_GRAY: u8 = 0b_0100_0000;

    /// 单通道灰色.
    pub const GRAY: u8 = 0b_1000_0000;

    /// 单通道亮灰色.
    pub const LIGHT_GRAY: u8 = 0b_1100_0000;

    /// 单通道白色.
    pub const WHITE: u8 = 0b_1111_1111;

    /// 像素是否是肿瘤?
    #[inline]
    pub const fn is_tumor(p: u8) -> bool {
        matches!(p, LITS_TUMOR)
    }

    /// 像素是否是肝脏?
    #[inline]
    pub const fn is_liver(p: u8) -> bool {
        matches!(p, LITS_LIVER)
    }

    /// 像素是否是背景?
    #[inline]
    pub const fn is_background(p: u8) -> bool {
        matches!(p, LITS_BACKGROUND)
    }

    /// 像素是否是边缘?
    #[inline]
    pub const fn is_boundary(p: u8) -> bool {
        matches!(p, LITS_BOUNDARY)
    }

    /// 像素是否是肝脏或边缘?
    #[inline]
    pub const fn is_liver_or_boundary(p: u8) -> bool {
        matches!(p, LITS_LIVER | LITS_BOUNDARY)
    }

    /// 像素是否是肝脏或肿瘤?
    #[inline]
    pub const fn is_liver_or_tumor(p: u8) -> bool {
        matches!(p, LITS_LIVER | LITS_TUMOR)
    }
}

/// LiTS 训练集大小.
pub const LITS_TRAINING_SET_LEN: u32 = 131;

/// LiTS 测试集大小.
pub const LITS_TESTING_SET_LEN: u32 = 70;

/// 体素/像素类型.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ElemType {
    /// `LITS_{LIVER, TUMOR}`, 代表前景.
    Background,

    /// `LITS_BACKGROUND`, 代表背景.
    Foreground,
}

impl ElemType {
    /// 是否为前景.
    #[inline]
    pub fn is_foreground(&self) -> bool {
        matches!(self, Self::Foreground)
    }

    /// 是否为背景.
    #[inline]
    pub fn is_background(&self) -> bool {
        !self.is_foreground()
    }
}
