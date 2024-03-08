//! 对 `ct-berry::dataset` 的更一层封装. 提供更直接的数据集加载器.

use ct_berry::dataset::lits_train::{self, CtDataLoader, LabelLoader};
use std::env;
use std::path::{Path, PathBuf};

/// 获取 LiTS 训练集标签基本路径.
///
/// 1. 若环境变量 `$LITS_TRAIN_LABEL_DIR` 非空, 则返回其值;
/// 2. 否则, 返回 `$HOME/dataset/train/label`.
pub fn label_dir_from_env_or_home() -> PathBuf {
    if let Ok(d) = env::var("LITS_TRAIN_LABEL_DIR") {
        PathBuf::from(d)
    } else {
        ct_berry::dataset::home_dataset_dir_with(["train", "label"]).unwrap()
    }
}

/// 获取 LiTS 训练集标签数据加载器.
pub fn label_loader<P: AsRef<Path>>(path: P) -> LabelLoader {
    lits_train::full_label_loader(path)
}

/// 从 `$LITS_TRAIN_LABEL_DIR` 或者 `$HOME/dataset/train/label` 下加载 LiTS 训练集标签基本路径.
#[inline]
pub fn label_loader_from_env_or_home() -> LabelLoader {
    label_loader(label_dir_from_env_or_home())
}

/// 获取 LiTS 训练集数据基本路径.
///
/// 1. 若环境变量 `$LITS_TRAIN_DIR` 非空, 则返回其值;
/// 2. 否则, 返回 `$HOME/dataset/train`.
pub fn train_dir_from_env_or_home() -> PathBuf {
    if let Ok(d) = env::var("LITS_TRAIN_DIR") {
        PathBuf::from(d)
    } else {
        ct_berry::dataset::home_dataset_dir_with(["train"]).unwrap()
    }
}

/// 获取 LiTS 训练集数据加载器.
pub fn data_loader<P: AsRef<Path>>(path: P) -> CtDataLoader {
    lits_train::full_ct_loader(path)
}

/// 从 `$LITS_TRAIN_LABEL_DIR` 或者 `$HOME/dataset/train` 下加载 LiTS 训练集数据加载器.
#[inline]
pub fn train_loader_from_env_or_home() -> CtDataLoader {
    data_loader(train_dir_from_env_or_home())
}
