//! 三次样条曲线.

use ndarray::{s, Array, Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::Solve;
use num::Float;
use std::ops::MulAssign;

// ref: https://zhuanlan.zhihu.com/p/628508199

macro_rules! impl_cubic {
    ($fp: ty, $one: expr, $two: expr, $three: expr) => {
        impl<'a> CubicSplineImp<'a, $fp> {
            #[inline]
            pub fn new(x: ArrayView1<'a, $fp>, y: ArrayView1<'a, $fp>, k: u32) -> Self {
                assert_eq!(x.len(), y.len(), "x 值和 y 值必须一一对应");
                assert!(x.len() >= 3, "该样条曲线至少需要三个点");
                assert!(k >= 10, "插值行为本身不应该使数据点过于稀疏");
                assert!(
                    x.windows(2).into_iter().all(|v| v[0] < v[1])
                        || x.windows(2).into_iter().all(|v| v[0] > v[1]),
                    "x 值必须严格递减或严格递增"
                );

                Self::new_no_check(x, y, k)
            }

            #[inline]
            fn new_no_check(x: ArrayView1<'a, $fp>, y: ArrayView1<'a, $fp>, k: u32) -> Self {
                Self { x, y, k }
            }

            pub fn make_spline(&self) -> (Vec<$fp>, Vec<$fp>) {
                // `k`: points per interval
                let len = self.x.len();
                let coe = self.spline_coefficient();

                let mut x1: Vec<$fp> = Vec::with_capacity(self.k as usize * (len - 1));
                let mut y1: Vec<$fp> = Vec::with_capacity(self.k as usize * (len - 1));

                for i in 0..(len - 1) {
                    let xs = Array::linspace(self.x[i], self.x[i + 1], self.k as usize + 1);
                    let mut dx = xs.clone();

                    // 栈上常访变量
                    let (mut rv1, rv2, rv3): ($fp, $fp, $fp);
                    rv1 = self.x[i];

                    dx.mapv_inplace(|v| v - rv1);

                    let mut ys = dx.clone();
                    (rv1, rv2, rv3) = (coe[(i, 2)], coe[(i, 1)], coe[(i, 0)]);
                    ys.mapv_inplace(|v| (v * rv1) + rv2);
                    ys.mul_assign(&dx.view());
                    ys.mapv_inplace(|v| v + rv3);
                    ys.mul_assign(&dx.view());
                    rv1 = self.y[i];
                    ys.mapv_inplace(|v| v + rv1);

                    x1.extend_from_slice(Self::array1_slice_except_last(xs.as_slice()));
                    y1.extend_from_slice(Self::array1_slice_except_last(ys.as_slice()));
                }
                x1.push(*self.x.last().unwrap());
                y1.push(*self.y.last().unwrap());

                (x1, y1)
            }

            fn array1_diff(arr: ArrayView1<$fp>) -> Array1<$fp> {
                let vector: Vec<$fp> = arr.windows(2).into_iter().map(|v| v[1] - v[0]).collect();
                Array1::from_vec(vector)
            }

            fn array1_slice_except_last(maybe_slice: Option<&[$fp]>) -> &[$fp] {
                let Some(&[ref s @ .., _]) = maybe_slice else {
                    unreachable!()
                };
                s
            }

            fn spline_coefficient(&self) -> Array2<$fp> {
                // assert!(x.len() >= 3);
                // assert_eq!(x.len(), y.len());

                let len = self.x.len();
                let mut a = Array2::<$fp>::zeros((len, len));
                let mut r = Array1::<$fp>::zeros(len);
                let dx = Self::array1_diff(self.x.view());
                let dy = Self::array1_diff(self.y.view());
                for i in 1..(len - 1) {
                    let mut a_slice = a.slice_mut(s!(i, (i - 1)..=(i + 1)));
                    a_slice.assign(&ArrayView1::from(&[
                        dx[i - 1],
                        $two * (dx[i - 1] + dx[i]),
                        dx[i],
                    ]));
                    r[i] = $three * (dy[i] / dx[i] - dy[i - 1] / dx[i - 1]);
                }
                *a.first_mut().unwrap() = $one;
                *a.last_mut().unwrap() = $one;

                let mut coe: Array2<$fp> = Array2::zeros((len, 3));

                let x = a.solve(&r).unwrap();
                coe.slice_mut(s!(.., 1)).assign(&x);

                for i in 0..(len - 1) {
                    coe[(i, 2)] = (coe[(i + 1, 1)] - coe[(i, 1)]) / ($three * dx[i]);
                    coe[(i, 0)] =
                        dy[i] / dx[i] - dx[i] * ($two * coe[(i, 1)] + coe[(i + 1, 1)]) / $three;
                }
                coe.remove_index(Axis(0), coe.len_of(Axis(0)) - 1);
                coe
            }
        }
    };
}

pub(crate) struct CubicSplineImp<'a, T: Float> {
    x: ArrayView1<'a, T>,
    y: ArrayView1<'a, T>,
    k: u32,
}

impl_cubic!(f32, 1.0_f32, 2.0_f32, 3.0_f32);
impl_cubic!(f64, 1.0_f64, 2.0_f64, 3.0_f64);
