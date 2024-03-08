//! 消融实验依赖的通用组件.

use ct_berry::CtWindow;

pub mod loader;

const SEP: &str = "--------------------------------------------------------";

/// 简单分隔线.
#[inline]
pub fn sep() {
    println!("{SEP}");
}

/// 简单分隔线.
#[inline]
pub fn sep_to<W: std::io::Write>(mut w: W) {
    writeln!(&mut w, "{SEP}").unwrap();
}

/// 获得可并行核心数.
pub fn cpus() -> usize {
    std::thread::available_parallelism().map_or_else(|_| num_cpus::get(), usize::from)
}

/// 创建一般情况下合适的、用于可视化腹部 CT 肝脏扫描的窗口.
/// 该窗口窗位为 60, 窗宽为 200.
#[inline]
pub fn liver_window() -> CtWindow {
    CtWindow::new(60.0, 200.0).unwrap()
}
