//! CT scan/label 切片对象的操作.

mod core;
mod iter;
mod mirror;
mod save;

pub use core::{
    CompactLabelSlice, LabelSlice, LabelSliceMut, OwnedLabelSlice, OwnedScanSlice, ScanSlice,
    ScanSliceMut,
};

pub use mirror::{LabelMirror, ScanMirror};

pub use save::{ImgWriteRaw, ImgWriteVis};

cfg_if::cfg_if! {
    if #[cfg(feature = "plot")] {
        mod plot;

        pub use plot::ImgDisplay;
    }
}
