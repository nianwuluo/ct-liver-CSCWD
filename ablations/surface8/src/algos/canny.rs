//! [canny 算法](https://ieeexplore.ieee.org/abstract/document/4767851).

use super::{Profile, Runtime};
use ct_berry::consts::gray::*;
use ct_berry::LabelSliceMut;
use opencv::core::{Mat, MatTraitConst, Scalar, CV_8U};
use opencv::imgproc;

impl<'a> Runtime<'a> {
    /// canny 算法分析.
    ///
    /// 注意该操作会修改原数据并消费自我 (consume `self`).
    /// 如有必要, 请提前做好备份.
    pub fn run_canny(mut self, profile: &mut Profile) -> LabelSliceMut<'a> {
        if !self.unify_binary() {
            profile.count_trivial();
            return self.consume();
        }

        profile.count_target(false);

        // 准备阶段, 不计入时长.
        // 先将原图替换为 255, 让 canny 算法能明显检测到边缘.
        self.replace(LITS_LIVER, WHITE);

        profile.target_start();

        let in_mat: Mat = self.make_owned_opencv_matrix();

        let mut out_mat: Mat =
            Mat::new_size_with_default(in_mat.size().unwrap(), CV_8U, Scalar::from(0)).unwrap();

        imgproc::canny(&in_mat, &mut out_mat, 50.0, 150.0, 3, false).unwrap();

        // 现在 `out_mat` 只存在两个值: BLACK 代表背景, WHITE 代表轮廓;
        // `self` 目前只存在两个值: BLACK (LITS_BACKGROUND) 代表背景, WHITE 代表肝脏.
        // 将 canny 算法结果回填到 `&mut self` 的原始数据.
        for x in 0..self.height() {
            for y in 0..self.width() {
                let out_val = *out_mat.at_2d::<u8>(x as i32, y as i32).unwrap();
                if out_val == WHITE {
                    self[(x, y)] = LITS_BOUNDARY;
                } else if self[(x, y)] == WHITE {
                    self[(x, y)] = LITS_LIVER;
                }
            }
        }

        // 为公平起见, 在计时前应当删除所有堆分配.
        drop(in_mat);
        drop(out_mat);

        profile.target_elapsed();

        self.consume()
    }
}
