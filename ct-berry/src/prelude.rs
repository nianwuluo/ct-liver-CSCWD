//! 🍇欢迎光临🍓
//!
//! 涵盖了本 crate 一系列常用的功能.

pub use crate::{Idx2d, Idx3d};

pub use crate::data::slice::{
    ImgWriteVis, LabelSlice, LabelSliceMut, OwnedLabelSlice, OwnedScanSlice, ScanSlice,
    ScanSliceMut,
};
pub use crate::data::window::CtWindow;
pub use crate::data::{CtData3d, CtLabel, CtScan, NiftiHeaderAttr};

#[cfg(feature = "plot")]
pub use crate::data::slice::ImgDisplay;

pub use crate::consts::gray::{LITS_BACKGROUND, LITS_BOUNDARY, LITS_LIVER, LITS_TUMOR};
pub use crate::consts::{ElemType, LITS_TESTING_SET_LEN, LITS_TRAINING_SET_LEN};

pub use crate::dataset::home_dataset_dir_with;
pub use crate::dataset::{self, lits_train};

pub use crate::sector::Sector;
