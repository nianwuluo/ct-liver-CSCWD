//! 水平切片镜像. 用于提取和复原 CT 切片.

use super::{LabelSlice, LabelSliceMut, ScanSlice, ScanSliceMut};

/// 一个拥有所有权的 CT 扫描水平切片的不透明镜像.
/// 用于临时保存一个水平标签切片的值，并在随后恢复.
#[derive(Clone, Debug)]
pub struct ScanMirror(pub(crate) Vec<f32>);

impl From<&ScanSlice<'_>> for ScanMirror {
    fn from(value: &ScanSlice<'_>) -> Self {
        Self(value.iter().copied().collect())
    }
}

impl From<&ScanSliceMut<'_>> for ScanMirror {
    fn from(value: &ScanSliceMut<'_>) -> Self {
        Self(value.iter().copied().collect())
    }
}

/// 一个拥有所有权的 CT 标签水平切片的不透明镜像.
/// 用于临时保存一个水平标签切片的值，并在随后恢复.
///
/// 注意该结构是被设计来 **快速** 回填原数据的,
/// 因此并不压缩原数据.
#[derive(Clone, Debug)]
pub struct LabelMirror(pub(crate) Vec<u8>);

impl From<&LabelSlice<'_>> for LabelMirror {
    fn from(value: &LabelSlice<'_>) -> Self {
        Self(value.iter().copied().collect())
    }
}

impl From<&LabelSliceMut<'_>> for LabelMirror {
    fn from(value: &LabelSliceMut<'_>) -> Self {
        Self(value.iter().copied().collect())
    }
}
