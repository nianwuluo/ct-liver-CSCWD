use std::collections::HashSet;

use binary_heap_plus::BinaryHeap;

use super::{idx2d_to_u16, idx3d_to_u16};
use crate::consts::gray::*;
use crate::{CtLabel, Idx2d, Idx3d, Idx3dU16, NiftiHeaderAttr};

#[inline]
fn idx3d_diff_pow((a, b, c): Idx3d, (x, y, z): Idx3d) -> (f64, f64, f64) {
    (
        a.abs_diff(x).pow(2) as f64,
        b.abs_diff(y).pow(2) as f64,
        c.abs_diff(z).pow(2) as f64,
    )
}

#[inline]
const fn is_foreground(include_tumor: bool, pixel: u8) -> bool {
    if include_tumor {
        is_liver_or_tumor(pixel)
    } else {
        is_liver(pixel)
    }
}

/// 实现提取 peripheral roi 所需要维护的相关数据结构.
pub struct RoiGenerator<'a> {
    label: &'a CtLabel,
    center: Idx3d,
}

// 3d 部分实现块
impl<'a> RoiGenerator<'a> {
    pub fn new(label: &'a CtLabel, &center: &Idx3d) -> Self {
        Self { label, center }
    }

    /// 以 3D 模式计算 `self.center` 到 `point` 的欧氏距离的平方, 单位为 (mm)^2.
    #[inline]
    fn center_distance_to_squared_3d(&self, point: &Idx3d) -> f64 {
        let (a, b, c) = idx3d_diff_pow(self.center, *point);
        let hm = self.label.height_mm().powi(2);
        let zm = self.label.z_mm().powi(2);
        (b + c) * hm + a * zm
    }

    /// 以 2D 模式计算 `self.center` 到 `point` 的欧氏距离的平方, 单位为 (mm)^2.
    #[inline]
    fn center_distance_to_squared_2d(&self, point: &Idx2d) -> f64 {
        let hm = self.label.slice_pixel();
        let a = self.center.1.abs_diff(point.0).pow(2) as f64;
        let b = self.center.2.abs_diff(point.1).pow(2) as f64;
        hm * (a + b)
    }

    /// 以 `self.center` 为中心, 提取半径不大于 `radius`
    /// (单位: mm) 的球的所有前景体素索引.
    pub fn extract_roi_3d(&self, radius: f64, include_tumor: bool) -> Vec<Idx3d> {
        // 堆顶距 `self.center` 最近
        let mut heap: BinaryHeap<Idx3d, _> = BinaryHeap::new_by(|a, b| {
            self.center_distance_to_squared_3d(b)
                .total_cmp(&self.center_distance_to_squared_3d(a))
        });

        heap.reserve(64);
        heap.push(self.center);
        let mut ans = Vec::with_capacity(64);
        let mut visited = HashSet::<Idx3dU16>::with_capacity(64);

        while let Some(pos) = heap.pop() {
            if self.center_distance_to_squared_3d(&pos) > radius.powi(2) {
                break;
            }
            let pos_u16 = idx3d_to_u16(&pos);
            if visited.contains(&pos_u16) {
                continue;
            }
            ans.push(pos);
            visited.insert(pos_u16);

            for dia_neigh in self.label.diamond_neighbours(pos) {
                let dia_u16 = idx3d_to_u16(&dia_neigh);
                if !visited.contains(&dia_u16) {
                    heap.push(dia_neigh);
                }
            }
        }
        ans.retain(|p| is_foreground(include_tumor, self.label[*p]));
        ans.shrink_to_fit();
        ans
    }

    /// 以 `self.center` 为中心, 提取半径不大于 `radius`
    /// (单位: mm) 的圆的所有前景体素索引.
    pub fn extract_roi_2d(&self, radius: f64, include_tumor: bool) -> Vec<Idx2d> {
        let (z, h, w) = self.center;

        let mut heap: BinaryHeap<Idx2d, _> = BinaryHeap::new_by(|a, b| {
            self.center_distance_to_squared_2d(b)
                .total_cmp(&self.center_distance_to_squared_2d(a))
        });

        let sli = self.label.slice_at(z);
        heap.reserve(32);
        heap.push((h, w));
        let mut ans = Vec::with_capacity(32);
        let mut visited = HashSet::with_capacity(32);

        while let Some(pos) = heap.pop() {
            if self.center_distance_to_squared_2d(&pos) > radius.powi(2) {
                break;
            }

            let pos_u16 = idx2d_to_u16(&pos);
            if visited.contains(&pos_u16) {
                continue;
            }
            visited.insert(pos_u16);
            ans.push(pos);

            for neigh in sli.n4_positions(pos) {
                let neigh_u16 = idx2d_to_u16(&neigh);
                if !visited.contains(&neigh_u16) {
                    heap.push(neigh);
                }
            }
        }
        ans.retain(|p| is_foreground(include_tumor, sli[*p]));
        ans.shrink_to_fit();
        ans
    }
}
