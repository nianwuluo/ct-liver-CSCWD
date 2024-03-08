//! LiTS CT scan 测试集数据加载器.
//!
//! 提供迭代器风格的扫描获取模式.

use crate::consts::LITS_TESTING_SET_LEN;
use crate::CtScan;
use std::path::{Path, PathBuf};

/// 从指定索引和路径创建 LiTS CT 测试集的 scans ([`CtScan`]) 加载器.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. `data` 的所有值 `value` 必须在 `path` 下有形如 `test-volume-{value}.nii` 的文件,
///   否则加载器在迭代时会返回 `Result::Error`.
pub fn test_loader<I: IntoIterator<Item = u32>, P: AsRef<Path>>(data: I, path: P) -> TestLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    let mut data: Vec<u32> = data.into_iter().collect();
    data.reverse();

    TestLoader {
        path,
        data_rev: data,
    }
}

/// 从指定路径创建 LiTS CT 测试集的 scans ([`CtScan`]) 加载器.
/// 返回的加载器会按索引序迭代 LiTS **测试集** 下所有的 CT scans.
///
/// # 注意
///
/// 1. `path` 必须是目录, 否则程序 panic.
/// 2. 对于 `0 <= value < crate::consts::LITS_TESTING_SET_LEN`, 必须在 `path`
///   下有形如 `test-volume-{value}.nii` 的文件, 否则加载器在迭代时会返回
///   `Result::Error`.
pub fn full_test_loader<P: AsRef<Path>>(path: P) -> TestLoader {
    let path = path.as_ref().to_owned();
    assert!(path.is_dir());

    TestLoader {
        path,
        data_rev: (0..LITS_TESTING_SET_LEN).rev().collect(),
    }
}

/// 3D CT scans 数据加载器.
#[derive(Debug)]
pub struct TestLoader {
    path: PathBuf,
    data_rev: Vec<u32>,
}

impl Iterator for TestLoader {
    type Item = (u32, nifti::Result<CtScan>);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.data_rev.pop()?;

        self.path.push(format!("test-volume-{idx}.nii"));
        let data = CtScan::open(self.path.as_path());
        self.path.pop();

        Some((idx, data))
    }
}
