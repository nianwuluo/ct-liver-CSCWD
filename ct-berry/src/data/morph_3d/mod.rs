//! 3D 形态学操作.

use self::phantom::PhantomMemento;
use self::roi::RoiGenerator;
use crate::consts::{gray::*, ElemType};
use crate::sector::{AxisDirection, Orientation};
use crate::{CtLabel, Idx2d, Idx2dU16, Idx3d, Idx3dU16, NiftiHeaderAttr};
use std::cmp::Ordering;
use std::ops::AddAssign;

mod phantom;

mod roi;

/// &Idx2d -> Idx2dU16
#[inline]
const fn idx2d_to_u16((h, w): &Idx2d) -> Idx2dU16 {
    // Usize to uShort
    (*h as u16, *w as u16)
}

/// &Idx3d -> Idx3dU16
#[inline]
const fn idx3d_to_u16((z, h, w): &Idx3d) -> Idx3dU16 {
    // Usize to uShort
    (*z as u16, *h as u16, *w as u16)
}

/// &Idx3dU16 -> Idx3d
#[inline]
const fn idx3du16_to_usize((z, h, w): &Idx3dU16) -> Idx3d {
    // uShort to Usize
    (*z as usize, *h as usize, *w as usize)
}

/// CT 2d 横切面上的垂直/水平单位向量类型.
#[derive(Copy, Clone, Debug)]
enum UnitVec {
    /// 2d 切片向下方向.
    HeightPos1,

    /// 2d 切片向上方向.
    HeightNeg1,

    /// 2d 切片向右方向.
    WidthPos1,

    /// 2d 切片向左方向.
    WidthNeg1,
}

impl UnitVec {
    /// 三维向量计算, 返回 `k * self + dataset`. 如果算数溢出则行为未定义.
    #[inline]
    pub fn add_to_k_3d(&self, data: &Idx3d, k: usize) -> Idx3d {
        match self {
            UnitVec::HeightPos1 => (data.0, data.1 + k, data.2),
            UnitVec::HeightNeg1 => (data.0, data.1 - k, data.2),
            UnitVec::WidthPos1 => (data.0, data.1, data.2 + k),
            UnitVec::WidthNeg1 => (data.0, data.1, data.2 - k),
        }
    }
}

impl AddAssign<UnitVec> for Idx3d {
    /// `self += rhs`. 如果算数溢出则行为未定义.
    #[inline]
    fn add_assign(&mut self, rhs: UnitVec) {
        match rhs {
            UnitVec::HeightPos1 => self.1 += 1,
            UnitVec::HeightNeg1 => self.1 -= 1,
            UnitVec::WidthPos1 => self.2 += 1,
            UnitVec::WidthNeg1 => self.2 -= 1,
        }
    }
}

impl AddAssign<UnitVec> for Idx2d {
    /// `self += rhs`. 如果算数溢出则行为未定义.
    #[inline]
    fn add_assign(&mut self, rhs: UnitVec) {
        match rhs {
            UnitVec::HeightPos1 => self.0 += 1,
            UnitVec::HeightNeg1 => self.0 -= 1,
            UnitVec::WidthPos1 => self.1 += 1,
            UnitVec::WidthNeg1 => self.1 -= 1,
        }
    }
}

/// Center ROI 实现块
impl CtLabel {
    /// 获取中心 ROI.
    ///
    /// 该函数对肝脏 CT label 进行三维形态学腐蚀, 找到形态学中心,
    /// 并以此为基准, 半径为 `radius` 获取周围所有肝脏 + 肿瘤体素.
    ///
    /// 如果指定 `anisotropic` 为 `true`,
    /// 则在三维腐蚀时会考虑体素本身的各向异性 (算法由该库提出并实现).
    /// 否则, 按照各向同性进行腐蚀. 如果 `include_tumor` 为 `true`,
    /// 则返回值不包含肿瘤索引. 实际上若 `include_tumor` 为 `false`,
    /// 则该函数将 `LITS_LIVER`, `LITS_TUMOR` 完全视为等价.
    ///
    /// # 注意
    ///
    /// 1. 允许存在肿瘤像素. 肿瘤像素在腐蚀中被当做肝脏像素处理.
    /// 2. 原扫描中不应当存在 `LITS_BACKGROUND` 背景空洞, 否则程序行为可能异常.
    ///   因此一般情况下请确保刚调用过 `self.fill_background_hollow()`.
    ///
    /// # 返回值
    ///
    /// 目标索引集合. 无顺序保证. 如果不存在前景, 则返回空 `Vec`.
    pub fn center_roi_3d(&self, radius: f64, anisotropic: bool, include_tumor: bool) -> Vec<Idx3d> {
        assert!(radius >= 0.0);
        let Some(center) = self.center(anisotropic) else {
            return vec![];
        };
        RoiGenerator::new(self, &center).extract_roi_3d(radius, include_tumor)
    }

    /// 与 [`Self::center_roi_3d`] 类似, 但在三维形态学腐蚀到中心后获取二维 ROI.
    /// 相应地, 返回值的第一个分量代表 ROI 所在的水平切片索引,
    /// 第二个分量是该水平切片上的二维索引集合. 如果不存在前景,
    /// 返回值元组的第一个分量为 `usize::MAX`, 第二个分量为空 `Vec`.
    pub fn center_roi_2d(
        &self,
        radius: f64,
        anisotropic: bool,
        include_tumor: bool,
    ) -> (usize, Vec<Idx2d>) {
        assert!(radius >= 0.0);
        let Some(center) = self.center(anisotropic) else {
            return (usize::MAX, vec![]);
        };
        (
            center.0,
            RoiGenerator::new(self, &center).extract_roi_2d(radius, include_tumor),
        )
    }

    /// 获取中心索引.
    ///
    /// 该函数对肝脏 CT label 进行形态学腐蚀, 直至到达中心. 如果指定
    /// `anisotropic` 为 `true`, 则则腐蚀时会考虑体素本身的各向异性
    /// (算法由该库提出并实现). 否则按照各向同性进行腐蚀.
    ///
    /// # 注意
    ///
    /// 1. 该函数将 `LITS_LIVER`, `LITS_TUMOR` 完全视为等价.
    ///   所以返回值索引也可能对应于 `LITS_TUMOR`.
    /// 2. 原扫描中不应当存在 `LITS_BACKGROUND` 背景空洞, 否则程序行为可能异常.
    ///   因此一般情况下请确保刚调用过 `self.fill_background_hollow()`.
    /// 3. 如果原图不存在 `LITS_{LIVER, TUMOR}` 则返回 `None`.
    ///
    /// # 返回值
    ///
    /// 中心点索引. 函数保证对相同扫描运行多次的结果一致 (稳定性).
    pub fn center(&self, anisotropic: bool) -> Option<Idx3d> {
        assert_eq!(self.height_mm(), self.width_mm());

        let (db, vox_cnt) = self.init_book_keeping();
        (vox_cnt != 0).then(|| {
            if anisotropic {
                self.center_roi_anisotropic(db, vox_cnt)
            } else {
                self.center_roi_isotropic(db, vox_cnt)
            }
        })
    }

    /// 遍历整个 3D 扫描, 返回适当初始化的 `PhantomMemento`
    /// 和目前肝脏 + 肿瘤总体素个数.
    fn init_book_keeping(&self) -> (PhantomMemento, usize) {
        let mut db = PhantomMemento::new();
        let mut vox_cnt = 0usize;

        for (pos, _) in self
            .data
            .indexed_iter()
            .filter(|(_, p)| is_liver_or_tumor(**p))
        {
            vox_cnt += 1;
            db.set_foreground(&pos); // <= 前景是 "肝脏 + 肿瘤"

            let mut is_surface = false; // 当前前景像素是否位于表面 (边缘)
            for neigh_pos in self.diamond_neighbours(pos) {
                if is_background(self[neigh_pos]) {
                    is_surface = true;
                    db.set_background(&neigh_pos);
                }
            }
            if is_surface {
                // 记录边缘体素
                db.push_pos(&pos);
            }
        }
        (db, vox_cnt)
    }

    /// 各向同性腐蚀到中心. 保证结果的稳定性.
    fn center_roi_isotropic(&self, mut db: PhantomMemento, mut vox_cnt: usize) -> Idx3d {
        debug_assert!(vox_cnt >= 1);

        loop {
            // fpos: Foreground position
            let fpos_set = db.take_positions();

            // 本轮循环要腐蚀的候选像素.
            let mut to_erode = Vec::with_capacity(64);

            // 全面 (钻石型) 腐蚀
            for fpos in fpos_set.iter().map(idx3du16_to_usize) {
                if db.is_visited(&fpos) {
                    continue;
                }
                debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));
                debug_assert!(vox_cnt >= 1);

                // 该前景像素加入候选集, 并在之后统一删除.
                db.set_visited(&fpos);
                to_erode.push(fpos);
                debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                for dia_pos in self.diamond_neighbours(fpos) {
                    debug_assert!(db.get_val(&dia_pos).is_some());
                    if !db.is_visited(&dia_pos) && db.get_val(&dia_pos).unwrap().is_foreground() {
                        db.push_pos_next(&dia_pos);
                    }
                }
            }
            if vox_cnt == to_erode.len() {
                debug_assert!(vox_cnt >= 1);
                break *to_erode.iter().min().unwrap();
            }

            // 本轮腐蚀
            for pos in to_erode.iter() {
                debug_assert!(db.get_val(pos).is_some_and(ElemType::is_foreground));
                db.set_background(pos);
                debug_assert!(db.get_val(pos).is_some_and(ElemType::is_background));
            }
            vox_cnt -= to_erode.len();
            db.step();
        }
    }

    /// 各向异性腐蚀到中心. 保证结果的稳定性.
    #[inline]
    fn center_roi_anisotropic(&self, db: PhantomMemento, vox_cnt: usize) -> Idx3d {
        debug_assert!(vox_cnt >= 1);

        let Some(o) = self.height_mm().partial_cmp(&self.z_mm()) else {
            unreachable!()
        };
        match o {
            Ordering::Greater => self.center_roi_with_height_greater(db, vox_cnt),
            Ordering::Less => self.center_roi_with_z_greater(db, vox_cnt),
            Ordering::Equal => self.center_roi_isotropic(db, vox_cnt),
        }
    }

    /// 各向异性腐蚀到中心. `height > z`.
    fn center_roi_with_height_greater(&self, mut db: PhantomMemento, mut vox_cnt: usize) -> Idx3d {
        debug_assert!(self.height_mm() > self.z_mm());

        let (step, barrier, mut cur_step) = (self.z_mm(), self.height_mm(), 0.0);
        // let mut iter_time = 1;
        loop {
            // 每一轮内循环需要查看所有前景表面候选像素, 并进行单次判定以决定该像素是否该被去除.
            // 如果不该去除, 就将其扔到 `db.surf2`; 否则就将其位置设置为 bg, 然后对其标记已访问.
            cur_step += step;
            let fpos_set = db.take_positions();

            // 本轮迭代要腐蚀的前景像素集合.
            let mut to_erode = Vec::with_capacity(64);
            debug_assert!(vox_cnt >= 1);

            if cur_step >= barrier {
                // 全面 (钻石型) 腐蚀.
                for fpos in fpos_set.iter().map(idx3du16_to_usize) {
                    if db.is_visited(&fpos) {
                        continue;
                    }
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    // 该前景像素加入候选集, 并在之后统一删除.
                    db.set_visited(&fpos);
                    to_erode.push(fpos);
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    for dia_pos in self.diamond_neighbours(fpos) {
                        debug_assert!(db.get_val(&dia_pos).is_some());
                        if !db.is_visited(&dia_pos) && db.get_val(&dia_pos).unwrap().is_foreground()
                        {
                            db.push_pos_next(&dia_pos);
                        }
                    }
                }
                cur_step -= barrier;
                debug_assert!(cur_step < barrier);
            } else {
                // only z 型腐蚀.
                for fpos in fpos_set.iter().map(idx3du16_to_usize) {
                    if db.is_visited(&fpos) {
                        continue;
                    }
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    let z_poss = self.neighbours_z(fpos);
                    if z_poss.iter().all(|p| db.is_foreground(p)) {
                        // z 方向被前景压住, 不符合腐蚀条件, 等待下一轮次
                        db.push_pos_next(&fpos);
                        continue;
                    }

                    // 仅将 z 方向腐蚀的表面前景像素集加入 visited.
                    db.set_visited(&fpos);
                    to_erode.push(fpos);
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    for dia_pos in self.diamond_neighbours(fpos) {
                        debug_assert!(db.get_val(&dia_pos).is_some());
                        if !db.is_visited(&dia_pos) && db.get_val(&dia_pos).unwrap().is_foreground()
                        {
                            db.push_pos_next(&dia_pos);
                        }
                    }
                }
            }

            if to_erode.len() == vox_cnt {
                break *to_erode.iter().min().unwrap();
            }
            // 本轮腐蚀
            for pos in to_erode.iter() {
                debug_assert!(db.get_val(pos).is_some_and(ElemType::is_foreground));
                db.set_background(pos);
                debug_assert!(db.is_visited(pos));
            }
            vox_cnt -= to_erode.len();
            db.step();
        }
    }

    /// 各向异性腐蚀到中心. `z > height`.
    fn center_roi_with_z_greater(&self, mut db: PhantomMemento, mut vox_cnt: usize) -> Idx3d {
        debug_assert!(self.z_mm() > self.height_mm());

        let (step, barrier, mut cur_step) = (self.height_mm(), self.z_mm(), 0.0);
        loop {
            // 每一轮内循环需要查看所有前景表面候选像素, 并进行单次判定以决定该像素是否该被去除.
            // 如果不该去除, 就将其扔到 `db.surf2`; 否则就将其位置设置为 bg, 然后对其标记已访问.
            cur_step += step;
            let fpos_set = db.take_positions();

            // 本轮迭代要腐蚀的前景像素集合.
            let mut to_erode = Vec::with_capacity(64);
            debug_assert!(vox_cnt >= 1);

            if cur_step >= barrier {
                // 全面 (钻石型) 腐蚀.
                for fpos in fpos_set.iter().map(idx3du16_to_usize) {
                    if db.is_visited(&fpos) {
                        continue;
                    }
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    // 该前景像素加入候选集, 并在之后统一删除.
                    db.set_visited(&fpos);
                    to_erode.push(fpos);
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    for dia_pos in self.diamond_neighbours(fpos) {
                        debug_assert!(db.get_val(&dia_pos).is_some());
                        if !db.is_visited(&dia_pos) && db.get_val(&dia_pos).unwrap().is_foreground()
                        {
                            db.push_pos_next(&dia_pos);
                        }
                    }
                }
                cur_step -= barrier;
                debug_assert!(cur_step < barrier);
            } else {
                // only h/w 型腐蚀.
                for fpos in fpos_set.iter().map(idx3du16_to_usize) {
                    if db.is_visited(&fpos) {
                        continue;
                    }
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    let hw_poss = self.neighbours_hw(fpos);
                    if hw_poss.iter().all(|p| db.is_foreground(p)) {
                        // 不符合腐蚀条件, 等待下一轮次
                        db.push_pos_next(&fpos);
                        continue;
                    }
                    // 仅将 h/w 方向腐蚀的表面前景像素集加入 visited.
                    db.set_visited(&fpos);
                    to_erode.push(fpos);
                    debug_assert!(db.get_val(&fpos).is_some_and(ElemType::is_foreground));

                    for dia_pos in self.diamond_neighbours(fpos) {
                        debug_assert!(db.get_val(&dia_pos).is_some());
                        if !db.is_visited(&dia_pos) && db.get_val(&dia_pos).unwrap().is_foreground()
                        {
                            db.push_pos_next(&dia_pos);
                        }
                    }
                }
            }

            if to_erode.len() == vox_cnt {
                break *to_erode.iter().min().unwrap();
            }
            // 本轮腐蚀
            for pos in to_erode.iter() {
                assert!(db.get_val(pos).is_some_and(ElemType::is_foreground));
                db.set_background(pos);
            }
            vox_cnt -= to_erode.len();
            db.step();
        }
    }
}

/// 3D Peripheral ROI 实现块
impl CtLabel {
    /// 获取 3 个外围 3D ROI.
    ///
    /// 该功能对肝脏 CT label 进行三维形态学腐蚀, 找到形态学中心,
    /// 并以此为基准, 按 posterior, lateral 和 anterior 三个方向
    /// 以 `alpha` 为比例进行扩张, 找到三个外围中心, 分别以 `radius`
    /// 为半径获取三个子区域的 **三维** ROI.
    ///
    /// 如果指定 `anisotropic` 为 `true`,
    /// 则在三维腐蚀时会考虑体素本身的各向异性 (算法由该库提出并实现).
    /// 否则, 按照各向同性进行腐蚀. 如果 `include_tumor` 为 `true`,
    /// 则返回值不包含肿瘤索引. 实际上若 `include_tumor` 为 `false`,
    /// 则该函数将 `LITS_LIVER`, `LITS_TUMOR` 完全视为等价.
    ///
    /// # 注意
    ///
    /// 1. 允许存在肿瘤像素. 肿瘤像素在腐蚀中被当做肝脏像素处理.
    /// 2. 原扫描中不应当存在 `LITS_BACKGROUND` 背景空洞, 否则程序行为可能异常.
    ///   因此一般情况下请确保刚调用过 `self.fill_background_hollow()`.
    ///
    /// # 返回值
    ///
    /// 返回值的顺序如下:
    ///
    /// 1. 前部 (anterior);
    /// 2. 后部 (posterior);
    /// 3. 侧面 (lateral).
    ///
    /// 每个 `Vec` 的顺序没有保证. 如果不存在前景, 则返回三个空 `Vec`.
    pub fn peripheral_roi_3d(
        &self,
        radius: f64,
        alpha: f64,
        anisotropic: bool,
        include_tumor: bool,
    ) -> [Vec<Idx3d>; 3] {
        assert!(radius >= 0.0);
        assert!((0.0..=1.0).contains(&alpha));

        let Some(center) = self.center(anisotropic) else {
            return Default::default();
        };
        self.get_three_circle_3d(center, radius, alpha, include_tumor)
    }

    /// 获取 3 个外围 3D ROI.
    ///
    /// 除了我们需要手动指定肝脏中心 `center` 之外,
    /// 该函数功能与 [`Self::peripheral_roi_3d`] 完全相同.
    ///
    /// # 注意
    ///
    /// 必须保证 `center` 不越界且其索引对应的体素值为肝脏, 否则程序 panic.
    pub fn peripheral_roi_3d_with_center(
        &self,
        center: Idx3d,
        radius: f64,
        alpha: f64,
        include_tumor: bool,
    ) -> [Vec<Idx3d>; 3] {
        assert!(self
            .data()
            .get(center)
            .is_some_and(|&p| is_liver_or_tumor(p)));
        assert!(radius >= 0.0);
        assert!((0.0..=1.0).contains(&alpha));

        self.get_three_circle_3d(center, radius, alpha, include_tumor)
    }

    /// 获取 3 个外围 ROI 中心.
    pub fn peripheral_centers(&self, center: Idx3d, alpha: f64) -> [Idx3d; 3] {
        self.get_peripheral_unit_vectors()
            .map(|d| self.get_peripheral_center_3d(&center, d, alpha))
    }

    /// 获取三个方向的外围 ROI, 以 3D 格式表示.
    #[inline]
    fn get_three_circle_3d(
        &self,
        center: Idx3d,
        radius: f64,
        alpha: f64,
        include_tumor: bool,
    ) -> [Vec<Idx3d>; 3] {
        let [d1, d2, d3] = self.get_peripheral_unit_vectors();
        let (c1, c2, c3) = (
            self.get_peripheral_center_3d(&center, d1, alpha),
            self.get_peripheral_center_3d(&center, d2, alpha),
            self.get_peripheral_center_3d(&center, d3, alpha),
        );
        // debug!("Anterior center: {c1:?}");
        // debug!("Posterior center: {c2:?}");
        // debug!("Lateral center: {c3:?}");
        [
            self.collect_liver_circle_3d(&c1, radius, include_tumor),
            self.collect_liver_circle_3d(&c2, radius, include_tumor),
            self.collect_liver_circle_3d(&c3, radius, include_tumor),
        ]
    }

    /// 根据 CT 3D 特征, 提取三个扩展方向. 返回的三个向量的模均为 1.
    ///
    /// # 返回值
    ///
    /// 返回值的顺序如下:
    ///
    /// 1. 前部 (anterior);
    /// 2. 后部 (posterior);
    /// 3. 侧面 (lateral).
    fn get_peripheral_unit_vectors(&self) -> [UnitVec; 3] {
        use {Orientation::*, UnitVec::*};

        let pattern = self.lls_sector_pattern().unwrap();
        match (pattern.quadrant(), pattern.orientation()) {
            (AxisDirection::HeightPos, CounterClockwise) => [HeightPos1, HeightNeg1, WidthNeg1],
            (AxisDirection::HeightNeg, Clockwise) => [HeightNeg1, HeightPos1, WidthNeg1],
            (AxisDirection::HeightPos, Clockwise) => [HeightPos1, HeightNeg1, WidthPos1],
            _ => unreachable!(),
        }
    }

    /// 给定 3D 初始点 `from` (`m`) 和单位向量 `direction` (`n`), 计算并返回
    /// `m + alpha * step * n`. 结果被四舍五入.
    ///
    /// 内部使用二维坐标计算.
    fn get_peripheral_center_3d(&self, from: &Idx3d, direction: UnitVec, alpha: f64) -> Idx3d {
        // 往前试探
        let (z, h, w) = *from;
        let sli = self.slice_at(z);
        let mut snake = (h, w);
        let mut step = 0usize;

        // 一步一步向前试探.
        while sli.get(snake).is_some_and(|&p| is_liver_or_tumor(p)) {
            snake += direction;
            step += 1;
        }

        direction.add_to_k_3d(from, (step as f64 * alpha).round() as usize)
    }

    /// 以 `center` 为中心, 收集体素体积总和为 `radius`
    /// (最接近的, 小于等于; 单位为立方毫米) 的体素索引集合.
    fn collect_liver_circle_3d(
        &self,
        center: &Idx3d,
        radius: f64,
        include_tumor: bool,
    ) -> Vec<Idx3d> {
        RoiGenerator::new(self, center).extract_roi_3d(radius, include_tumor)
    }
}

/// 2D Peripheral ROI 实现块
impl CtLabel {
    /// 获取 3 个外围 2D ROI.
    ///
    /// 该功能对肝脏 CT label 进行三维形态学腐蚀, 找到形态学中心,
    /// 并以此为基准, 按 posterior, lateral 和 anterior 三个方向
    /// 以 `alpha` 为比例进行扩张, 找到三个外围中心, 分别以 `radius`
    /// 为半径获取三个子区域的 **平面** ROI.
    ///
    /// 如果指定 `anisotropic` 为 `true`,
    /// 则在三维腐蚀时会考虑体素本身的各向异性 (算法由该库提出并实现).
    /// 否则, 按照各向同性进行腐蚀. 如果 `include_tumor` 为 `true`,
    /// 则返回值不包含肿瘤索引. 实际上若 `include_tumor` 为 `false`,
    /// 则该函数将 `LITS_LIVER`, `LITS_TUMOR` 完全视为等价.
    ///
    /// # 注意
    ///
    /// 1. 允许存在肿瘤像素. 肿瘤像素在腐蚀中被当做肝脏像素处理.
    /// 2. 原扫描中不应当存在 `LITS_BACKGROUND` 背景空洞, 否则程序行为可能异常.
    ///   因此一般情况下请确保刚调用过 `self.fill_background_hollow()`.
    ///
    /// # 返回值
    ///
    /// 第一个分量为三个 ROI 所在的水平切片索引, 第二个分量的顺序为:
    ///
    /// 1. 前部 (anterior);
    /// 2. 后部 (posterior);
    /// 3. 侧面 (lateral).
    ///
    /// 每个 `Vec` 的顺序没有保证.
    /// 如果不存在前景, 则第一个分量为 `usize::MAX`, 第二个分量为三个空 `Vec`.
    pub fn peripheral_roi_2d(
        &self,
        radius: f64,
        alpha: f64,
        anisotropic: bool,
        include_tumor: bool,
    ) -> (usize, [Vec<Idx2d>; 3]) {
        assert!(radius >= 0.0);
        assert!((0.0..=1.0).contains(&alpha));

        let Some(center) = self.center(anisotropic) else {
            return (usize::MAX, Default::default());
        };
        (
            center.0,
            self.get_three_circle_2d(center, radius, alpha, include_tumor),
        )
    }

    /// 获取 3 个外围 2D ROI.
    ///
    /// 除了我们需要手动指定肝脏中心 `center` 之外,
    /// 该函数功能与 [`Self::peripheral_roi_2d`] 完全相同.
    ///
    /// # 注意
    ///
    /// 1. 必须保证 `center` 不越界且其索引对应的体素值为肝脏, 否则程序 panic.
    /// 2. 返回值不像 [`Self::peripheral_roi_2d`] 那样包含第一个分量, 因为它就是 `center.0`.
    pub fn peripheral_roi_2d_with_center(
        &self,
        center: Idx3d,
        radius: f64,
        alpha: f64,
        include_tumor: bool,
    ) -> [Vec<Idx2d>; 3] {
        assert!(self
            .data()
            .get(center)
            .is_some_and(|&p| is_liver_or_tumor(p)));
        assert!(radius >= 0.0);
        assert!((0.0..=1.0).contains(&alpha));

        self.get_three_circle_2d(center, radius, alpha, include_tumor)
    }

    /// 获取三个方向的外围 ROI, 以 2D 格式表示.
    #[inline]
    fn get_three_circle_2d(
        &self,
        center: Idx3d,
        radius: f64,
        alpha: f64,
        include_tumor: bool,
    ) -> [Vec<Idx2d>; 3] {
        let [d1, d2, d3] = self.get_peripheral_unit_vectors();

        // 这里仍然用 3D 的模式 (内部是 2D 实现).
        let (c1, c2, c3) = (
            self.get_peripheral_center_3d(&center, d1, alpha),
            self.get_peripheral_center_3d(&center, d2, alpha),
            self.get_peripheral_center_3d(&center, d3, alpha),
        );

        [
            self.collect_liver_circle_2d(&c1, radius, include_tumor),
            self.collect_liver_circle_2d(&c2, radius, include_tumor),
            self.collect_liver_circle_2d(&c3, radius, include_tumor),
        ]
    }

    /// 以 `center.0` 为水平切片索引, 以 `(center.1, center.2)` 为中心,
    /// 在该切片上收集半径不大于 `radius` (最接近的, 小于等于; 单位为平方毫米)
    /// 的所有像素索引集合.
    fn collect_liver_circle_2d(
        &self,
        center: &Idx3d,
        radius: f64,
        include_tumor: bool,
    ) -> Vec<Idx2d> {
        RoiGenerator::new(self, center).extract_roi_2d(radius, include_tumor)
    }
}
