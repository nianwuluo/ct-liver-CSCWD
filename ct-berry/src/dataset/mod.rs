//! 数据集操作.

use std::path::{Path, PathBuf};

pub mod generic;
pub mod lits_test;
pub mod lits_train;
mod npz_database;

pub use npz_database::{NpzArchive, OpenArchiveError};

/// 获取 `{用户主目录}/dataset` 目录.
pub fn home_dataset_dir() -> Option<PathBuf> {
    let mut ans = dirs::home_dir()?;
    ans.push("dataset");
    Some(ans)
}

/// 获取 `{用户主目录}/dataset` 目录下给定继续项组成的全路径.
pub fn home_dataset_dir_with<P: AsRef<Path>, I: IntoIterator<Item = P>>(it: I) -> Option<PathBuf> {
    let mut ans = dirs::home_dir()?;
    ans.push("dataset");
    ans.extend(it);
    Some(ans)
}
