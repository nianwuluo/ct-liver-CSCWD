//! 算法运行统计.

use std::time::{Duration, Instant};

/// ablation/benchmark 计时器.
///
/// 该计时器支持 "中途中断" 与 "结束中断, 继续开始计时".
#[derive(Clone, Debug)]
struct AccTimer {
    consumed: Duration,
    since: Instant,
}

impl AccTimer {
    /// 初始化计时器. 初始化时会视为已经开始计时 (`self.start()`).
    /// 如果用户不希望这种行为, 可以在真正需要时重新调用 `self.start()` 覆盖之.
    #[inline]
    pub fn new() -> Self {
        Self {
            consumed: Duration::from_secs(0),
            since: Instant::now(),
        }
    }

    /// 开始计时. 可以通过反复调用来重置, 或者通过之后的 `self.elapsed()`
    /// 方法来统计该部分时间.
    #[inline]
    pub fn start(&mut self) {
        self.since = Instant::now();
    }

    /// 结束计时, 并将这一区间的时间累加. 返回本轮计时时长.
    ///
    /// # 注意
    ///
    /// 上一次调用必须是 `self.start()`, 否则计算时间值无意义.
    #[inline]
    pub fn elapsed(&mut self) -> Duration {
        let d = self.since.elapsed();
        self.consumed += d;
        d
    }

    /// 获得总共累计下来的时间综合 (以微秒为单位).
    #[inline]
    pub fn get_total_us(&self) -> u64 {
        self.consumed.as_micros() as u64
    }

    /// 获得总共累计下来的时间综合 (以毫秒为单位).
    #[inline]
    #[allow(dead_code)]
    pub fn get_total_ms(&self) -> u64 {
        self.consumed.as_millis() as u64
    }

    /// 获得总共累计下来的时间综合 (以纳秒为单位).
    #[inline]
    #[allow(dead_code)]
    pub fn get_total_ns(&self) -> u128 {
        self.consumed.as_nanos()
    }
}

impl Default for AccTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// ablation/benchmark 数据统计.
#[derive(Clone, Debug)]
pub struct Profile {
    /// 唯一化处理后, 遇到的纯背景图片个数.
    trivial: u64,

    /// 遇到的普通图片个数 (除 `trivial` 以外的所有图片均符合要求).
    target: u64,

    /// 处理普通图片花费的总时间 (包括 CPU 时间, 系统 IO/调度时间).
    target_time: AccTimer,

    /// 整个任务花费的总时间 (包括 CPU 时间, 系统 IO/调度时间, 配置外部环境时间).
    real_time: AccTimer,

    /// 处理普通图片最耗时的一次任务所消耗的时间.
    most: Duration,

    /// 在具有唯一大肝脏对象的图片上运行算法损失的总肝脏像素 (体素) 个数.
    eroded: u64,
}

impl Profile {
    /// 初始化.
    #[inline]
    pub fn new() -> Self {
        Self {
            trivial: 0,
            target: 0,
            target_time: AccTimer::default(),
            real_time: AccTimer::default(),
            most: Duration::MAX,
            eroded: 0,
        }
    }

    /// 记录一个新的全背景图像.
    #[inline]
    pub fn count_trivial(&mut self) {
        self.trivial += 1;
    }

    /// 记录一个普通 (即带肝脏) 图像. `start` 表明是否同时开启新一轮图像处理计时任务.
    #[inline]
    pub fn count_target(&mut self, start: bool) {
        self.target += 1;
        if start {
            self.target_start();
        }
    }

    /// 开始一次新的普通图像处理计时.
    #[inline]
    pub fn target_start(&mut self) {
        self.target_time.start();
    }

    // pub fn target_pause(&mut self) { ... }

    /// 结束一次普通图像处理计时.
    #[inline]
    pub fn target_elapsed(&mut self) {
        let d = self.target_time.elapsed();
        self.most = match self.most {
            Duration::MAX => d,
            once_duration => std::cmp::max(d, once_duration),
        };
    }

    /// 如果 `count` 不为 0, 则添加一次腐蚀记录.
    #[inline]
    pub fn count_eroded(&mut self, count: u64) {
        self.eroded += count;
    }

    /// 结束全部计时.
    #[inline]
    pub fn finish(mut self) -> Self {
        self.real_time.elapsed();
        self
    }

    /// 获得总腐蚀记录.
    #[inline]
    pub fn get_eroded(&self) -> u64 {
        self.eroded
    }

    /// 获得总纯背景图像个数.
    #[inline]
    pub fn get_trivial(&self) -> u64 {
        self.trivial
    }

    /// 获得总普通图像个数.
    #[inline]
    pub fn get_target(&self) -> u64 {
        self.target
    }

    /// 以微秒为单位获得处理普通图像的总花费自然时间.
    #[inline]
    pub fn get_target_time_us(&self) -> u64 {
        self.target_time.get_total_us()
    }

    /// 以微秒为单位获得算法运行到目前的总自然时间.
    #[inline]
    pub fn get_real_time_us(&self) -> u64 {
        self.real_time.get_total_us()
    }

    /// 以毫秒为单位获得处理普通图像的平均时间.
    #[inline]
    pub fn get_avg_target_time_us(&self) -> Option<f64> {
        match self.target {
            0 => None,
            target => Some(self.get_target_time_us() as f64 / target as f64),
        }
    }

    /// 获得处理普通图像的平均腐蚀值.
    #[inline]
    pub fn get_avg_eroded(&self) -> Option<f64> {
        match self.target {
            0 => None,
            target => Some(self.get_eroded() as f64 / target as f64),
        }
    }

    /// 获取处理普通图片最耗时的一次任务所消耗的时间.
    ///
    /// 如果不存在任务, 则返回 `None`.
    pub fn get_most_time_consuming(&self) -> Option<Duration> {
        match self.most {
            Duration::MAX => None,
            d => Some(d),
        }
    }
}

impl Default for Profile {
    fn default() -> Self {
        Self::new()
    }
}
