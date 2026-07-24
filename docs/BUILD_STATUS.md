# Build status

Completed in this developer package:

- frozen frame ROI and Stage 1–3 model weights included
- final geometry-aware pipeline scripts included
- Windows-safe Python orchestration
- CPU/GPU device selection
- editable SQLite project
- cell add/delete/move/class-change GUI
- automatic Excel, CSV, crop-overlay, and original-image-overlay export
- CPU and CUDA environment installers
- PyInstaller build scripts
- Linux-reference regression validator

Not completed in this environment:

- a compiled Windows `.exe`
- execution of the regression test on Windows
- Windows code signing
- final external-distribution license decision

PyInstaller must be executed on Windows. The resulting CPU/GPU ZIP must not be
treated as a release until the included regression test passes.
