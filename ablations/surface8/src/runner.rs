//! 程序运行函数.

use crate::result::AblationResult;
use std::thread;
use utils::loader;

/// 实际运行.
pub fn run() -> AblationResult {
    let label_dir = loader::label_dir_from_env_or_home();
    assert!(label_dir.is_dir());
    let p = label_dir.as_path();

    // 短路判断
    assert!(
        loader::label_loader(p)
            .next()
            .is_some_and(|(_, r)| r.is_ok()),
        "Loading dataset config error"
    );

    println!("Running ablation studies...");
    thread::scope(|s| {
        use super::algos::*;

        let handles = [canny, suzuki, hrvoje, mulberry].map(|t| s.spawn(move || t(p)));

        AblationResult::from_iter(
            ["canny", "suzuki", "hrvoje", "mulberry"].into_iter().zip(
                handles
                    .into_iter()
                    .map(|th| th.join().expect("Thread joining error")),
            ),
        )
    })
}
