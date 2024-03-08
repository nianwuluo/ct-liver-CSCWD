use super::{Profile, Runtime};
use ct_berry::consts::gray::*;
use ct_berry::prelude::*;
use opencv::core::{Mat, MatTraitConst, Point, Scalar, CV_8U};
use opencv::imgproc::{self, CHAIN_APPROX_NONE, LINE_8, RETR_EXTERNAL};
use opencv::types::VectorOfVectorOfPoint;

impl<'a> Runtime<'a> {
    /// suzuki 算法分析.
    ///
    /// 注意该操作会修改原数据并消费自我 (consume `self`).
    /// 如有必要, 请提前做好备份.
    pub fn run_suzuki85(mut self, profile: &mut Profile) -> LabelSliceMut<'a> {
        if !self.unify_binary() {
            profile.count_trivial();
            return self.consume();
        }

        profile.count_target(true);

        let in_mat: Mat = self.make_owned_opencv_matrix();

        let mut contours = VectorOfVectorOfPoint::with_capacity(1);
        let mut contours_img =
            Mat::new_size_with_default(in_mat.size().unwrap(), CV_8U, Scalar::from(0)).unwrap();

        imgproc::find_contours(
            &in_mat,
            &mut contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_NONE,
            Point::new(0, 0),
        )
        .unwrap();
        imgproc::draw_contours(
            &mut contours_img,
            &contours,
            -1,
            Scalar::from(LITS_BOUNDARY as i32),
            1,
            LINE_8,
            &opencv::core::no_array(),
            i32::MAX,
            Point::new(0, 0),
        )
        .unwrap();

        // 将 suzuki 算法结果回填到 `&mut self` 的原始数据.
        for x in 0..self.height() {
            for y in 0..self.width() {
                let out_val = *contours_img.at_2d::<u8>(x as i32, y as i32).unwrap();
                if out_val == LITS_BOUNDARY {
                    self[(x, y)] = LITS_BOUNDARY;
                }
            }
        }

        // 为公平起见, 在计时前应当删除所有堆分配.
        drop(contours);
        drop(contours_img);

        profile.target_elapsed();

        self.consume()
    }
}
