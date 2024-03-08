//! 2D 肝脏 CT 水平切片的边缘优化 (后处理).

use crate::consts::{gray::*, ElemType};
use crate::sector::Sector;
use crate::{Idx2d, LabelSliceMut, ScanSlice};
use std::collections::{HashSet, VecDeque};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 对前景表面实施优化.
///
/// 该算法假设 `label` 的元素仅包括 `LITS_BACKGROUND`, `LITS_LIVER` 和
/// `LITS_TUMOR` 三类 (不假设任何其它性质). 其中第一种像素为背景,
/// 后两种像素统称前景.
///
/// 算法流程依次为:
///
/// 1. 按照 4-相邻规则定位所有前景像素.
/// 2. 从第一步的所有像素出发, 进行 `bfs_step` 步 BFS, 获取所有遍历到的像素
///   (包括第一步的前景像素本身).
/// 3. 联合肝衰减 `attenuation` 和 `threshold` (HU) 优化门限,
///   对每个上一步收集到的所有像素的位置, 参考 `scan` 的对应位置.
///   其中落在 `threshold` 范围内的像素被修改为 `LITS_LIVER`,
///   其它像素被修改为 `LITS_BACKGROUND`.
/// 4. 提取在 `Sector` 扇区范围内以如以上算法被修改的所有位置详细信息.
///
/// # 注意
///
/// - 如果 `label` 违反上述规定, 则程序行为未定义;
/// - 如果 `scan` 和 `label` 不能相互对应, 则程序行为未定义 (可能 panic).
/// - `attenuation` 和 `threshold` 描述的是纯粹的 CT HU 值, 与窗口无关.
pub fn refine_surface<'a>(
    scan: ScanSlice<'a>,
    label: LabelSliceMut<'a>,
    bfs_step: u32,
    attenuation: f64,
    threshold: HuThreshold,
    sector: Sector,
) -> Refined {
    RefineImp::new(scan, label, bfs_step, attenuation, threshold, sector).refine()
}

/// 同 `refine_surface`, 但不返回对修改结果的描述.
pub fn refine_surface0<'a>(
    scan: ScanSlice<'a>,
    label: LabelSliceMut<'a>,
    bfs_step: u32,
    attenuation: f64,
    threshold: HuThreshold,
    sector: Sector,
) {
    RefineImp::new(scan, label, bfs_step, attenuation, threshold, sector).refine0();
}

/// CT HU 值优化门限. 用于决定一个 CT HU
/// 值应该被优化为前景 (特指肝脏像素) 还是背景.
/// 由用户负责确保对象的值合法.
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum HuThreshold {
    /// \[liver_attenuation - value, liver_attenuation + value\]
    /// 闭区间内的 CT HU 值被视为肝脏部分, 在此以外的被视为背景。
    Centered(f64),

    /// \[value, +inf) 左闭右开区间内的
    /// CT HU 值被视为肝脏部分, 在此以外的被视为背景.
    GreaterThan(f64),
}

impl HuThreshold {
    /// 给定肝衰减 `attenuation` 和某扫描位置体素/像素值 `pix_val`,
    /// 判断该位置被优化后的像素类型.
    pub fn eval(&self, attenuation: f64, pix_val: f64) -> ElemType {
        match *self {
            HuThreshold::Centered(half) => {
                let r = (attenuation - half)..=(attenuation + half);
                if r.contains(&pix_val) {
                    ElemType::Foreground
                } else {
                    ElemType::Background
                }
            }
            HuThreshold::GreaterThan(th) => {
                if pix_val >= th {
                    ElemType::Foreground
                } else {
                    ElemType::Background
                }
            }
        }
    }
}

/// 肝脏表面 refine 的方式.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Modified {
    /// 原先是前景像素, 在 refine 过程中被修改为背景像素.
    F2B,

    /// 原先是背景像素, 在 refine 过程中被修改为前景
    /// (这里特指肝脏) 像素.
    B2F,
}

/// 封装了所有被 refine 的 (索引, refine 方式) 元组.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Refined {
    // 使用 `Vec<(Idx2d, RefinedManner)` 会导致大量 padding,
    // 故使用两个 `Vec` 分离存储.
    data: Vec<Idx2d>,
    how: Vec<Modified>,
}

impl Refined {
    /// 获得元素个数.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 判断是否为空.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 获得第 `index` 个分量.
    #[inline]
    pub fn at(&self, index: usize) -> (Idx2d, Modified) {
        (self.data[index], self.how[index])
    }

    /// 获取能迭代全部修改的迭代器.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (Idx2d, Modified)> + '_ {
        self.data.iter().copied().zip(self.how.iter().copied())
    }

    /// 直接获得内部数据的所有权.
    #[inline]
    pub fn into_raw(self) -> (Vec<Idx2d>, Vec<Modified>) {
        (self.data, self.how)
    }

    /// 内部方法, 初始化.
    #[inline]
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(4),
            how: Vec::with_capacity(4),
        }
    }

    /// 内部方法, 添加元素.
    #[inline]
    fn push(&mut self, index: Idx2d, how: Modified) {
        self.data.push(index);
        self.how.push(how);
    }
}

/// `refine_surface`, `refine_surface0` 函数的实现细节.
struct RefineImp<'a> {
    scan: ScanSlice<'a>,
    label: LabelSliceMut<'a>,
    bfs_step: u32,
    attenuation: f64,
    threshold: HuThreshold,
    sector: Sector,
}

impl<'a> RefineImp<'a> {
    #[inline]
    pub fn new(
        scan: ScanSlice<'a>,
        label: LabelSliceMut<'a>,
        bfs_step: u32,
        attenuation: f64,
        threshold: HuThreshold,
        sector: Sector,
    ) -> Self {
        assert_eq!(scan.shape(), label.shape());
        Self {
            scan,
            label,
            bfs_step,
            attenuation,
            threshold,
            sector,
        }
    }

    fn bfs(&self) -> HashSet<Idx2d> {
        // step 1: 按照 4-相邻规则定位所有符合要求的前景/背景像素.
        let mut q = self.get_init_surface();
        if q.is_empty() {
            return Default::default();
        }

        // step 2: 从第一步的所有像素出发, 进行 `bfs_step` 步 BFS,
        // 获取所有遍历到的像素 (包括预处理时的表面像素).
        let mut visited: HashSet<Idx2d> = HashSet::with_capacity(q.len());
        for _ in 0..self.bfs_step {
            if q.is_empty() {
                break;
            }
            let len = q.len();
            for _ in 0..len {
                let cur = q.pop_front().unwrap();
                if visited.contains(&cur) {
                    continue;
                }
                visited.insert(cur);
                q.extend(
                    self.label
                        .n4_positions(cur)
                        .iter()
                        .filter(|p| !visited.contains(*p)),
                );
            }
        }

        visited
    }

    /// 运行实际优化, 并返回修改明细.
    #[inline]
    pub fn refine(&mut self) -> Refined {
        // step 3: 以 `threshold` (HU) 为门限,
        //   对每个上一步收集到的所有像素的位置,
        //   参考 `scan` 的对应位置, 并在需要时进行优化.
        let mut delta = Refined::new();
        for pos in self.bfs().into_iter().filter(|p| self.sector.contains(*p)) {
            let ct_hu = self.scan[pos] as f64;
            let orig_pixel = self.label[pos];
            if ElemType::Foreground == self.threshold.eval(self.attenuation, ct_hu) {
                self.label[pos] = LITS_LIVER;
                if is_background(orig_pixel) {
                    delta.push(pos, Modified::B2F);
                }
            } else {
                self.label[pos] = LITS_BACKGROUND;
                if is_liver(orig_pixel) {
                    delta.push(pos, Modified::F2B);
                }
            }
        }

        delta
    }

    /// 运行实际优化.
    #[inline]
    pub fn refine0(&mut self) {
        // step 3: 以 `threshold` (HU) 为门限,
        //   对每个上一步收集到的所有像素的位置,
        //   参考 `scan` 的对应位置, 并在需要时进行优化.
        for pos in self.bfs().into_iter().filter(|p| self.sector.contains(*p)) {
            let ct_hu = self.scan[pos] as f64;
            if ElemType::Foreground == self.threshold.eval(self.attenuation, ct_hu) {
                self.label[pos] = LITS_LIVER;
            } else {
                self.label[pos] = LITS_BACKGROUND;
            }
        }
    }

    /// 按照 4-相邻规则定位所有表面像素.
    ///
    /// 满足以下任意条件之一的像素被称为表面像素:
    ///
    /// 1. 前景像素, 且 4-邻域包含背景像素;
    /// 2. 背景像素, 且 4-邻域包含前景像素.
    fn get_init_surface(&self) -> VecDeque<Idx2d> {
        self.label
            .array_view()
            .indexed_iter()
            .filter_map(|(index, &pixel)| {
                ((is_liver_or_tumor(pixel) && self.label.is_n4_containing_background(index))
                    || (is_background(pixel) && self.label.is_n4_having(index, is_liver_or_tumor)))
                .then_some(index)
            })
            .collect()
    }
}
