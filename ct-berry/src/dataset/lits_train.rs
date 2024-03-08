//! LiTS CT scan/label 训练集数据加载器.
//!
//! 提供迭代器风格的数据集获取模式.

use crate::consts::LITS_TRAINING_SET_LEN;
use crate::{CtData3d, CtLabel, CtScan};
use std::path::{Path, PathBuf};

/// 从指定索引和路径创建 LiTS CT 训练集的 scans ([`CtScan`]) 加载器.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. `data` 的所有值 `value` 必须在 `path` 下有形如 `volume-{value}.nii` 的文件,
///   否则加载器在迭代时会返回 `Result::Error`.
pub fn scan_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(data: I, path: P) -> ScanLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    ScanLoader {
        path,
        data_rev: data,
    }
}

/// 从指定路径创建 LiTS CT 训练集的 scans ([`CtScan`]) 加载器.
/// 返回的加载器会按索引序迭代 LiTS **训练集** 下所有的 CT scans.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. 对于 `0 <= value < crate::consts::LITS_TRAINING_SET_LEN`, 必须在 `path`
///   下有形如 `volume-{value}.nii` 的文件, 否则加载器在迭代时会返回
///   `Result::Error`.
pub fn full_scan_loader<P: AsRef<Path>>(path: P) -> ScanLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    ScanLoader {
        path,
        data_rev: (0..LITS_TRAINING_SET_LEN).rev().collect(),
    }
}

/// 3D CT scans 数据加载器.
#[derive(Debug)]
pub struct ScanLoader {
    path: PathBuf,
    data_rev: Vec<u32>,
}

impl Iterator for ScanLoader {
    type Item = (u32, nifti::Result<CtScan>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.path.push(format!("volume-{idx}.nii"));
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

/// 从指定索引和路径创建 LiTS CT 训练集的 labels ([`CtLabel`]) 加载器.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. `data` 的所有值 `value` 必须在 `path` 下有形如 `segmentation-{value}.nii`
///   的文件, 否则加载器在迭代时会返回 `Result::Error`.
pub fn label_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(data: I, path: P) -> LabelLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    LabelLoader {
        path,
        data_rev: data,
    }
}

/// 从指定路径创建 LiTS CT 训练集的 labels ([`CtLabel`]) 加载器.
/// 返回的加载器会按索引序迭代 LiTS **训练集** 下所有的 CT labels.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. 对于 `0 <= value < crate::consts::LITS_TRAINING_SET_LEN`, 必须在 `path`
///   下有形如 `segmentation-{value}.nii` 的文件, 否则加载器在迭代时会返回
///   `Result::Error`.
pub fn full_label_loader<P: AsRef<Path>>(path: P) -> LabelLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    LabelLoader {
        path,
        data_rev: (0..LITS_TRAINING_SET_LEN).rev().collect(),
    }
}

/// 3D CT labels 数据加载器.
#[derive(Debug)]
pub struct LabelLoader {
    path: PathBuf,
    data_rev: Vec<u32>,
}

impl Iterator for LabelLoader {
    type Item = (u32, nifti::Result<CtLabel>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.path.push(format!("segmentation-{idx}.nii"));
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

/// 假设 `path` 是数据集目录.
/// 返回值: (scan path, label path)
/// 注意: `path/scan` 或 `path/label` 目录不存在时 panic.
#[inline]
fn make_loader_path<P: AsRef<Path>>(path: P) -> (PathBuf, PathBuf) {
    let mut scan_path = path.as_ref().to_owned();
    let mut label_path = scan_path.clone();
    scan_path.push("scan");
    assert!(scan_path.is_dir());

    label_path.push("label");
    assert!(label_path.is_dir());

    (scan_path, label_path)
}

/// 从指定路径创建 LiTS CT 训练集的数据 ([`CtData3d`]) 加载器.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. `path` 下必须存在 "scan" 目录和 "label" 目录, 否则程序 panic.
/// 3. `data` 的所有值 `value` 必须在 "scan" 和 "label" 目录下分别存在形如
///   `volume-{value}.nii` 和 `segmentation-{value}.nii` 的文件,
///   否则加载器在迭代时会返回 `Result::Error`.
/// 4. 相同索引对应的 volume 和 segmentation 必须一一对应, 否则程序行为未定义.
pub fn ct_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(
    data: I,
    dataset_path: P,
) -> CtDataLoader {
    let (scan_path, label_path) = make_loader_path(dataset_path);
    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    CtDataLoader {
        scan_path,
        label_path,
        data_rev: data,
    }
}

/// 从指定路径创建 LiTS CT 训练集的数据 ([`CtData3d`]) 加载器.
/// 返回的加载器会按索引序迭代 LiTS **训练集** 下的所有 CT data.
///
/// # 注意
///
/// 1. `dataset_path` 必须是目录, 并且目录下存在 "scan" 和 "label"
///   目录, 否则程序 panic.
/// 2. 对于 `0 <= value < crate::consts::LITS_TRAINING_SET_LEN`, 必须在
///   "scan" 和 "label" 目录下分别存在形如 `volume-{value}.nii` 和
///   `segmentation-{value}.nii` 的文件, 否则加载器在迭代时会返回 `Result::Error`.
/// 3. 相同索引对应的 volume 和 segmentation 必须一一对应, 否则程序行为未定义.
pub fn full_ct_loader<P: AsRef<Path>>(dataset_path: P) -> CtDataLoader {
    let (scan_path, label_path) = make_loader_path(dataset_path);

    CtDataLoader {
        scan_path,
        label_path,
        data_rev: (0..LITS_TRAINING_SET_LEN).rev().collect(),
    }
}

/// 3D CT 数据集 (scan + label) 加载器.
#[derive(Debug)]
pub struct CtDataLoader {
    scan_path: PathBuf,
    label_path: PathBuf,
    data_rev: Vec<u32>,
}

impl Iterator for CtDataLoader {
    type Item = (u32, nifti::Result<CtData3d>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.scan_path.push(format!("volume-{idx}.nii"));
        self.label_path.push(format!("segmentation-{idx}.nii"));
        let data = CtData3d::open(&self.scan_path, &self.label_path);
        self.label_path.pop();
        self.scan_path.pop();

        Some((idx, data))
    }
}

impl ExactSizeIterator for CtDataLoader {
    #[inline]
    fn len(&self) -> usize {
        self.data_rev.len()
    }
}
