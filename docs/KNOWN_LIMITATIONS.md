# Known limitations in developer v1

- The Windows executables have not yet been built or regression-tested on a
  Windows computer.
- The GUI currently loads the 5,000-pixel masked frame crop for editing.
- Undo/redo across previous sessions is not yet exposed in the GUI, although
  all edits are retained in `Edit_History`.
- Only images placed directly in the selected input folder are analyzed.
- Very large batches require substantial disk space because intermediate
  pipeline files are retained for reproducibility.
- CPU inference is expected to be much slower than NVIDIA GPU inference.
- The current PyInstaller specification may need additional hidden imports
  after the first Windows build, depending on the exact package versions.
