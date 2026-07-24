# GeoComb user guide

## New analysis

1. Open GeoComb.
2. Select `Analyze folder`.
3. Choose a folder containing frame images.
4. Choose an empty project folder.
5. Select `cpu`, `cuda`, or `auto`.
6. Wait until all nine pipeline stages complete.

## Editing

- Left click a circle to select it.
- Drag a selected circle to move the cell center.
- Double-click an empty location to add a cell.
- Press `1`–`8` to assign a class:
  1. Egg
  2. Larvae
  3. Capped_pupae
  4. Pollen
  5. Capped_honey
  6. Uncapped_honey
  7. Empty_cell
  8. Unbuilt_cell
- Press Delete to remove selected cells.
- Use the mouse wheel to zoom.
- Hold the middle mouse button to pan.

The selected class also becomes the class used for newly added cells.

## Automatic saving

Every edit is immediately written to `project.geocomb`. After a short delay,
GeoComb regenerates:

- `results/GeoComb_results.xlsx`
- the current image overlays
- image-specific cell CSV files

When the Excel workbook is open in Microsoft Excel, Windows may prevent it
from being replaced. GeoComb then writes
`GeoComb_results_pending_update.xlsx`. Close Excel and select `Export now`.

## Coordinate systems

The editor uses the 5,000-pixel masked frame crop because it is the exact
coordinate system used by the classifier and frozen overlay workflow.

The Excel file contains both:

- 5,000-pixel frame-crop coordinates
- coordinates mapped back to the original input image

## Project portability

Move the entire project folder together. Do not move only the Excel file,
because the SQLite database and pipeline images are also required.
