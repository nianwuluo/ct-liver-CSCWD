# Eight-neighborhood sparse contour extraction in binary images

The document gives brief information about how to run ablation studies.
Implementation details can be fully viewed in this workspace, so we
will not unfold them concretely in the document.

## Requirements / Prerequisite

1. [Install rust](https://www.rust-lang.org/learn/get-started). Make sure that `version >= 1.75.0`.
2. [Install opencv and set proper environment variables](https://crates.io/crates/opencv).
3. Prepare LiTS dataset. There are ways to place your dataset though (see codes), we
   strongly recommend that you put all segmentation label files into `$HOME/dataset/train/label`,
   which should be named `segmentation-{num}.nii` (`0` &lt;= num &lt;= `130`).

# Run

You can run ablation studies with the following command:

```shell
cargo run --release --package surface8
```

# Note

1. There are codes for unrelated (and well, not that private) research in `ct-berry` package.
   You can just ignore them for convenience. Make use of code intelligence
   (like rust-analyzer provided in visual studio code) to track down stuffs
   you really care about.
2. We call our proposed algorithm "Mulberry" in codes.
3. The way we run the experiment is to have each algorithm have a separate (native) thread.
 Therefore, if your computer has a small number of CPU cores (say, no more than 4), you
 can consider modifying the program to sequential one to get more accurate time information.

