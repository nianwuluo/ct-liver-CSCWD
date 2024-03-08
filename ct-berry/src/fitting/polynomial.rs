//! 多项式曲线.

// ref: https://blog.csdn.net/u012494154/article/details/112519550

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Inverse;

struct Polynomial<T: num::Float>(Array1<T>);

macro_rules! impl_polynomial {
    ($fp: ty, $zero: expr) => {
        impl Polynomial<$fp> {
            pub fn eval(&self, x: $fp) -> $fp {
                self.0.iter().rev().fold($zero, |acc, &cur| acc * x + cur)
            }
        }
    };
}

//// Advanced:
// macro_rules! macro0 {
//     () => {};
//     ($fp: ty, $($rest: ty),*) => {
//         /* define structures... */
//         macro0!($($rest),*);
//     };
// }
impl_polynomial!(f32, 0.0_f32);
impl_polynomial!(f64, 0.0_f64);

pub(crate) struct PolyImp<'a, T: num::Float> {
    x: ArrayView1<'a, T>,
    y: ArrayView1<'a, T>,
    degree: u32,
    points: u32,
    minmax: (T, T),
}

macro_rules! impl_poly_imp {
    ($fp: ty) => {
        impl<'a> PolyImp<'a, $fp> {
            /// `degree` 是多项式次数, `points` 是测试点个数, 会在区间内等距拟合.
            pub fn new(
                x: ArrayView1<'a, $fp>,
                y: ArrayView1<'a, $fp>,
                degree: u32,
                points: u32,
            ) -> Self {
                assert_eq!(x.len(), y.len(), "x 值和 y 值必须一一对应");
                assert!(x.len() >= 2, "至少需要拟合两个点");
                assert_ne!(degree, 0, "拟合曲线的次数不能为 0");
                assert!(points >= 3, "至少需要获得三个自变量 x.");

                Self {
                    x,
                    y,
                    degree,
                    points,
                    minmax: Self::min_max(x),
                }
            }

            pub fn make_curve(&self) -> (Vec<$fp>, Vec<$fp>) {
                let v_mat = self.vandermonde();
                let v_mat_t = v_mat.t();

                let theta = v_mat
                    .t()
                    .dot(&v_mat)
                    .inv()
                    .unwrap()
                    .dot(&v_mat_t)
                    .dot(&self.y);

                debug_assert_eq!(theta.len(), (self.degree + 1) as usize);

                let poly = Polynomial(theta);
                let step = self.step();
                let mut input = self.minmax.0;
                let mut ans_x = Vec::with_capacity(self.points as usize);
                let mut ans_y = Vec::with_capacity(self.points as usize);

                ans_x.push(input);
                ans_y.push(poly.eval(input));
                for _ in 0..self.points {
                    input += step;
                    ans_x.push(input);
                    ans_y.push(poly.eval(input));
                }

                (ans_x, ans_y)
            }

            fn vandermonde(&self) -> Array2<$fp> {
                // shape: (m, n); m = x.len(), n = self.degree + 1
                Array2::<$fp>::from_shape_fn((self.x.len(), self.degree as usize + 1), |(m, n)| {
                    self.x[m].powi(n as i32)
                })
            }

            #[inline]
            fn step(&self) -> $fp {
                (self.minmax.1 - self.minmax.0) / (self.points - 1) as $fp
            }

            fn min_max(arr: ArrayView1<$fp>) -> ($fp, $fp) {
                // !arr.is_empty()
                let (mut min_val, mut max_val) = (<$fp>::MAX, <$fp>::MIN);
                for v in arr.iter().copied() {
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
                (min_val, max_val)
            }
        }
    };
}

impl_poly_imp!(f32);
impl_poly_imp!(f64);
