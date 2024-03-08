use ndarray::{Array3, Ix3, OwnedRepr};
use ndarray_npy::{NpzReader, ReadNpzError};
use std::fs::{File, OpenOptions};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// 打开 `NpzArchive` 错误.
#[derive(Debug)]
pub enum OpenArchiveError {
    /// workers 太大. 最多支持 64.
    TooManyWorkers(u32),

    /// 打开 npz 文件错误.
    ReadNpzError(ReadNpzError),

    /// 其他底层 I/O 错误.
    IoError(std::io::Error),
}

/// Npz 文件归档.
///
/// 该结构可用于建模硬盘上已存储的多个 3D CT labels 的压缩文件.
pub struct NpzArchive {
    entries: Vec<Mutex<NpzReader<File>>>,
    turn: AtomicUsize,
}

impl NpzArchive {
    /// 初始化.
    ///
    /// `workers` 指定了底层工作通道的个数, 最大为 64. 系统会从路径 `p` 打开文件
    /// `workers` 次, 并为每个打开通道指定一个排他入口点 (以期获得更高的并行度).
    pub fn new<P: AsRef<Path>>(workers: NonZeroUsize, p: P) -> Result<Self, OpenArchiveError> {
        let workers = workers.get();
        if workers > 64 {
            return Err(OpenArchiveError::TooManyWorkers(64));
        }
        let mut v = Vec::with_capacity(workers);
        for _ in 0..workers {
            let file = OpenOptions::new()
                .read(true)
                .open(p.as_ref())
                .map_err(OpenArchiveError::IoError)?;
            v.push(Mutex::new(
                NpzReader::new(file).map_err(OpenArchiveError::ReadNpzError)?,
            ));
        }
        Ok(Self {
            entries: v,
            turn: AtomicUsize::new(0),
        })
    }

    /// 通过 npz 索引文件名 `name` 获取底层 3D 标签内容.
    pub fn label_by_name(&self, name: &str) -> Result<Array3<u8>, ReadNpzError> {
        let slot = self.next_slot();
        let mut file = self.entries[slot].lock().unwrap();
        file.by_name::<OwnedRepr<u8>, Ix3>(name)
    }

    /// 通过文件名 `{num}.npy` 获取底层 3D 标签内容.
    pub fn label_by_num_dot_npy(&self, num: u32) -> Result<Array3<u8>, ReadNpzError> {
        let slot = self.next_slot();
        let filename = format!("{num}.npy");
        let mut file = self.entries[slot].lock().unwrap();
        file.by_name::<OwnedRepr<u8>, Ix3>(filename.as_str())
    }

    /// 获取底层 npz 文件包含的所有文件名.
    pub fn label_names(&self) -> Result<Vec<String>, ReadNpzError> {
        let slot = self.next_slot();
        self.entries[slot].lock().unwrap().names()
    }

    /// 通过 npz 数值索引获取底层 3D 标签内容.
    pub fn label_by_index(&self, index: usize) -> Result<Array3<u8>, ReadNpzError> {
        let slot = self.next_slot();
        let mut file = self.entries[slot].lock().unwrap();
        file.by_index::<OwnedRepr<u8>, Ix3>(index)
    }

    /// 工作通道个数.
    #[inline]
    pub fn worker_len(&self) -> usize {
        self.entries.len()
    }

    /// 获取底层 npz 文件的 3D labels 个数.
    pub fn label_len(&self) -> usize {
        let slot = self.next_slot();
        self.entries[slot].lock().unwrap().len()
    }

    fn next_slot(&self) -> usize {
        self.turn.fetch_add(1, Ordering::Relaxed) % self.worker_len()
    }
}
