# Third-party notices

GeoComb is distributed as a fully open-source project under the GNU Affero
General Public License, version 3 (`AGPL-3.0-only`). The complete
corresponding source, build scripts, configuration files, and model
documentation are provided in the public GeoComb repository.

Third-party components remain the property of their respective copyright
holders and remain subject to their own licenses. The release process must
generate a package-version inventory from the actual CPU and CUDA build
environments and retain applicable license and notice files with the
distributed application.

## Ultralytics

GeoComb uses Ultralytics software and models under the AGPL-3.0 open-source
route. The GeoComb application is therefore released with its complete
corresponding source under AGPL-3.0.

Official licensing information:
https://www.ultralytics.com/license

## PySide6 / Qt for Python

GeoComb uses PySide6 and Qt for Python. The open-source Qt/PySide components
used by the packaged application remain under their applicable LGPL/GPL and
third-party terms. Packaged releases must preserve required notices and
license texts and must not impose restrictions that conflict with the
applicable open-source terms.

Official licensing information:
https://doc.qt.io/qtforpython-6/licenses.html

## Other software dependencies

GeoComb bundles or depends on software including PyTorch, OpenCV, NumPy,
SciPy, pandas, openpyxl, Pillow, and PyInstaller. The exact package names,
versions, metadata, and detected license files must be regenerated from
each release environment. Inclusion in an AGPL-licensed application does
not replace the license terms of these independent components.

PyInstaller's bootloader exception permits distribution of bundled
applications, subject to the licenses of the bundled dependencies.

## Model and dataset rights

- GeoComb model weights are released only after confirming that the
  authors are authorized to distribute them.
- DeepBee and FAIRHiveFrames-1K raw images are not redistributed by
  GeoComb. Users must obtain them from their original sources.
- Public repository manifests may identify evaluated images without
  republishing third-party image content.
- In-house images and annotations are distributed only to the extent
  permitted by the responsible institution and contributors.
- Demonstration images included in a binary or source release must have
  explicit redistribution permission.

## Release archive requirements

Each binary release should include:

- the GeoComb `LICENSE` file;
- this `THIRD_PARTY_NOTICES.md` file;
- the generated package-version and license inventory;
- available license or notice files collected from the actual environment;
  and
- a direct link to the corresponding tagged source release.

This document records the project's open-source distribution approach and
is not legal advice. Institutional review may still be required before
public release.
