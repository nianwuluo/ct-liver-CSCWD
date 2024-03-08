//! 通用 CT scan/label 数据加载器.
//!
//! 提供迭代器风格的数据集获取模式.

use crate::{CtData3d, CtLabel, CtScan};
use std::path::{Path, PathBuf};

/// 文件名构造器. 接受数据集索引数, 获得文件名.
pub type FilenameBuilder = fn(u32) -> String;

/// 从指定索引、路径、文件名构造器来创建通用的 CT scans 加载器.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. `data` 的所有取值 `value` 必须在 `path` 下有形如 `builder(value)` 的 nifti
///   文件, 否则加载器在迭代时会返回 `Result::Error`.
pub fn scan_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(
    data: I,
    path: P,
    builder: FilenameBuilder,
) -> ScanLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    ScanLoader {
        path,
        data_rev: data,
        builder,
    }
}

/// 3D CT scans 数据加载器, 并在内部自动转换文件名.
pub struct ScanLoader {
    path: PathBuf,
    data_rev: Vec<u32>,
    builder: FilenameBuilder,
}

impl Iterator for ScanLoader {
    type Item = (u32, nifti::Result<CtScan>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.path.push((self.builder)(idx));
        let data = CtScan::open(self.path.as_path());
        self.path.pop();

        Some((idx, data))
    }
}

impl ExactSizeIterator for ScanLoader {
    #[inline]
    fn len(&self) -> usize {
        self.data_rev.len()
    }
}

/// 从指定索引、路径、文件名构造器来创建通用的 CT labels 加载器.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. `data` 的所有取值 `value` 必须在 `path` 下有形如 `builder(value)` 的 nifti
///   文件, 否则加载器在迭代时会返回 `Result::Error`.
pub fn label_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(
    data: I,
    path: P,
    builder: FilenameBuilder,
) -> LabelLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    LabelLoader {
        path,
        data_rev: data,
        builder,
    }
}

/// 3D CT labels 数据加载器, 并在内部自动转换文件名.
#[derive(Debug)]
pub struct LabelLoader {
    path: PathBuf,
    data_rev: Vec<u32>,
    builder: FilenameBuilder,
}

impl Iterator for LabelLoader {
    type Item = (u32, nifti::Result<CtLabel>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.path.push((self.builder)(idx));
        let data = CtLabel::open(self.path.as_path());
        self.path.pop();

        Some((idx, data))
    }
}

impl ExactSizeIterator for LabelLoader {
    #[inline]
    fn len(&self) -> usize {
        self.data_rev.len()
    }
}

/// 从指定索引、路径、文件名构造器来创建通用的 CT data 加载器.
///
/// # 注意
///
/// 1. `scan_path` 和 `label_path` 必须是目录, 否则程序 panic.
/// 2. `data` 的所有取值 `value` 必须在 `scan_path` 下有形如
///   `scan_builder(value)` 的 nifti 文件, 否则加载器在迭代时会返回 `Result::Error`.
/// 3. `data` 的所有取值 `value` 必须在 `label_path` 下有形如
///   `label_builder(value)` 的 nifti 文件, 否则加载器在迭代时会返回 `Result::Error`.
pub fn data_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(
    data: I,
    scan_path: P,
    scan_builder: FilenameBuilder,
    label_path: P,
    label_builder: FilenameBuilder,
) -> CtDataLoader {
    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    let scan_path = scan_path.as_ref().to_owned();
    let label_path = label_path.as_ref().to_owned();

    CtDataLoader {
        scan_path,
        scan_builder,
        label_path,
        label_builder,
        data_rev: data,
    }
}

/// 3D CT 数据集 (scan + label) 加载器, 并在内部自动转换文件名.
#[derive(Debug)]
pub struct CtDataLoader {
    scan_path: PathBuf,
    scan_builder: FilenameBuilder,
    label_path: PathBuf,
    label_builder: FilenameBuilder,
    data_rev: Vec<u32>,
}

impl Iterator for CtDataLoader {
    type Item = (u32, nifti::Result<CtData3d>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.scan_path.push((self.scan_builder)(idx));
        self.label_path.push((self.label_builder)(idx));
        let data = CtData3d::open(&self.scan_path, &self.label_path);
        self.label_path.pop();
        self.scan_path.pop();

        Some((idx, data))
    }
}
