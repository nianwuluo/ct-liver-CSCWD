/// CT 窗口, 包含窗位 (window level) 和窗宽 (window width).
///
/// 该窗口是只读的. 若要修改窗口参数, 你应该创建新的实例.
#[derive(Copy, Clone, Debug)]
pub struct CtWindow {
    level: f32,
    width: f32,
}

impl CtWindow {
    /// 构建 CT 窗.
    ///
    /// `level` 和 `width` 必须在合理范围内, 否则返回 `None`.
    pub fn new(level: f32, width: f32) -> Option<CtWindow> {
        if (-1e5..=1e5).contains(&level) && 0.0 < width && width <= 1e5 {
            Some(Self { level, width })
        } else {
            None
        }
    }

    /// 构建一个便于展示扫描图像肝脏结构的 CT 窗口. 该窗口的窗位为
    /// 60, 窗宽为 200.
    #[inline]
    pub const fn from_liver_visual() -> CtWindow {
        Self {
            level: 60.0,
            width: 200.0,
        }
    }

    /// 窗下限.
    #[inline]
    pub fn lower_bound(&self) -> f32 {
        self.level - self.width / 2.0
    }

    /// 窗上限.
    #[inline]
    pub fn upper_bound(&self) -> f32 {
        self.level + self.width / 2.0
    }

    /// 窗位.
    #[inline]
    pub fn level(&self) -> f32 {
        self.level
    }

    /// 窗宽.
    #[inline]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// 求在当前 CT 窗设置下, `ct` HU 值对应的灰度图像素整数值 (0 <= value <= 255)
    ///
    /// 如果 `ct` 无意义 (如 inf, NaN), 则返回 `None`.
    pub fn eval(&self, ct: f32) -> Option<u8> {
        if !ct.is_finite() {
            return None;
        }
        let lb = self.lower_bound();
        if ct <= lb {
            Some(u8::MIN)
        } else if ct >= self.upper_bound() {
            Some(u8::MAX)
        } else {
            // 255, not 256.
            Some((((ct - lb) / self.width()) * 255.0) as u8)
        }
    }

    /// 求在当前 CT 窗设置下, `ct` HU 值对应的灰度图像素分布点 (0.0 <= value <= 255.0).
    ///
    /// 如果 `ct` 无意义 (如 inf, NaN), 则返回 `None`.
    pub fn eval_f32(&self, ct: f32) -> Option<f32> {
        if !ct.is_finite() {
            return None;
        }
        let lb = self.lower_bound();
        let ub = self.upper_bound();
        if ct <= lb {
            Some(0.0)
        } else if ct >= ub {
            Some(255.0)
        } else {
            Some((ct - lb) / self.width() * 255.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::CtWindow;

    fn is_valid_init(level: f32, width: f32) -> bool {
        CtWindow::new(level, width).is_some()
    }

    #[test]
    fn test_ct_window_invalid_input() {
        assert!(!is_valid_init(0.0, -1.0));
        assert!(!is_valid_init(0.0, 0.0));
    }

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-8
    }

    #[test]
    fn test_ct_window_generic() {
        // [60, 100]
        let ct = CtWindow::new(80.0, 40.0).unwrap();
        assert_eq!(ct.eval(f32::NAN), None);
        assert_eq!(ct.eval(f32::MIN), Some(0));
        assert_eq!(ct.eval(f32::MAX), Some(255));

        assert_eq!(ct.eval(50.0), Some(0));
        assert!(float_eq(ct.eval_f32(50.0).unwrap(), 0.0));

        assert_eq!(ct.eval(60.0), Some(0));
        assert!(float_eq(ct.eval_f32(60.0).unwrap(), 0.0));

        // boundary 1
        assert_eq!(ct.eval(60.1), Some(0));
        assert!(ct.eval_f32(60.1).unwrap() > 0.0);
        assert!(ct.eval_f32(60.1).unwrap() < 1.0);
        // -- boundary 1

        assert_eq!(ct.eval(70.0).unwrap(), (255.0 * 0.25) as u8);
        assert!(float_eq(ct.eval_f32(70.0).unwrap(), 255.0 * 0.25));

        assert_eq!(ct.eval(80.0).unwrap(), (255.0 * 0.5) as u8);
        assert!(float_eq(ct.eval_f32(80.0).unwrap(), 255.0 * 0.5));

        assert_eq!(ct.eval(90.0).unwrap(), (255.0 * 0.75) as u8);
        assert!(float_eq(ct.eval_f32(90.0).unwrap(), 255.0 * 0.75));

        // boundary 2
        assert_eq!(ct.eval(99.999), Some(254));
        assert!(ct.eval_f32(99.999).unwrap() < 255.0);
        assert!(ct.eval_f32(99.999).unwrap() > 254.0);
        // -- boundary 2

        assert_eq!(ct.eval(100.0).unwrap(), u8::MAX);
        assert!(float_eq(ct.eval_f32(100.0).unwrap(), 255.0));
    }
}
