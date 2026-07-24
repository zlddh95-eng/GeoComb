# GeoComb user guide

## 1. Choose a Windows package

GeoComb is distributed as separate CPU and NVIDIA CUDA packages.

- Use `GeoComb_CPU.exe` on a supported Windows x64 computer.
- Use `GeoComb_GPU.exe` only on a computer with a compatible NVIDIA GPU and
  driver.

Extract the entire downloaded archive. Do not move the executable away
from the accompanying `_internal` folder or other packaged files.

Expected executable locations:

```text
GeoComb_Windows_CPU/GeoComb_CPU/GeoComb_CPU.exe
GeoComb_Windows_GPU/GeoComb_GPU/GeoComb_GPU.exe
```

If Windows displays a security warning for an unsigned research
application, verify that the archive came from the official GeoComb
release and that its SHA-256 hash matches the published release manifest
before continuing.

## 2. Start a new analysis

1. Open the appropriate GeoComb executable.
2. Select `Analyze Folder`.
3. Choose a folder containing the frame images to analyze.
4. Choose a new or empty project folder.
5. If a device option is displayed, retain the package-appropriate default.
6. Wait until the analysis is complete.

Only supported image files placed directly in the selected input folder
are analyzed. Nested subfolders are not traversed.

## 3. Open an existing project

1. Open GeoComb.
2. Select `Open Project`.
3. Choose the project folder or its `project.geocomb` file.

Opening an existing project does not rerun the models. The editable project
database remains the authoritative record.

## 4. Edit detected cells

- Left-click a circle to select it.
- Drag a selected circle to move the cell center.
- Double-click an empty location to add a cell.
- Press `Delete` to remove selected cells.
- Use the mouse wheel to zoom.
- Hold the middle mouse button to pan.
- Press `1`-`8` to assign a class:

  1. Egg
  2. Larvae
  3. Capped pupae
  4. Pollen
  5. Capped honey
  6. Uncapped honey
  7. Empty cell
  8. Unbuilt cell

The selected class becomes the class used for newly added cells.

## 5. Automatic saving and export

Every edit is written to `project.geocomb`. After a short delay, GeoComb
regenerates:

- `results/GeoComb_results.xlsx`;
- current image overlays; and
- image-specific cell CSV files.

If the Excel workbook is open in Microsoft Excel, Windows may prevent it
from being replaced. GeoComb then writes
`GeoComb_results_pending_update.xlsx`. Close Excel and select `Export Now`
to regenerate the main workbook.

## 6. Coordinate systems

The editor uses the 5,000-pixel masked frame crop because this is the
coordinate system used by the classifier and frozen overlay workflow. The
Excel workbook contains both:

- 5,000-pixel frame-crop coordinates; and
- coordinates mapped back to the original input image.

## 7. Project portability

Move or archive the entire project folder. Do not move only the Excel
workbook because the SQLite project database, source metadata, and
pipeline images are also required.

Before transferring a project, close GeoComb and Microsoft Excel to ensure
that all files have been released by Windows.

## 8. Reproducibility and support information

Record the following when reporting a problem:

- GeoComb release tag and package type;
- Windows version;
- CPU or GPU model;
- NVIDIA driver version for CUDA use;
- input-image filename;
- project log; and
- whether the input or project folder is on a local, network, or cloud
  drive.

Do not upload confidential images or project files to a public issue.
