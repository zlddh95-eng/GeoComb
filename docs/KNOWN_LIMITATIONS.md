# Known limitations of the GeoComb platform release

- Platform validation was performed on one AMD Ryzen 9 7950X / NVIDIA RTX
  5070 Ti workstation using Windows and WSL2 Linux. Compatibility and
  performance on broader hardware have not yet been established.
- The Windows CUDA package requires a compatible NVIDIA GPU and driver.
- The editor displays the 5,000-pixel masked frame crop used by the
  classifier and maps saved coordinates back to the original input image.
- Undo/redo across previous sessions is not exposed in the GUI, although
  edits are retained in the project history.
- Only supported images located directly in the selected input folder are
  analyzed; nested folders are not traversed.
- Large batches require substantial disk space because intermediate files
  are retained for reproducibility and expert review.
- CPU analysis is substantially slower than NVIDIA CUDA analysis.
- Performance may decrease for image domains, frame types, viewpoints,
  illumination conditions, or cell appearances that are not represented in
  the reported benchmarks.
- Rare cell classes remain more difficult than common capped-brood,
  honey-storage, and empty-comb categories.
- GeoComb outputs are research estimates that should be reviewed by a
  knowledgeable user when used outside the validated image domains.
- The current evaluation used a limited number of fully annotated frames.
  Some references were initialized from model predictions before complete
  manual correction, so residual annotation anchoring cannot be excluded.
- GeoComb does not provide colony-health diagnosis or management
  recommendations.
