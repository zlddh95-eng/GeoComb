from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QKeySequence,
    QPen,
    QPixmap,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    CLASS_COLORS_RGB,
    CLASS_ORDER,
    CLASS_SHORTCUTS,
)
from .database import ProjectDatabase
from .excel_export import export_excel
from .overlay_export import export_image_overlays
from .pipeline import PipelineRunner


class AnalysisWorker(QObject):
    log = Signal(str)
    progress = Signal(str, int, int)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        input_folder: Path,
        project_folder: Path,
        device: str,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.project_folder = project_folder
        self.device = device

    def run(self) -> None:
        try:
            runner = PipelineRunner(
                log_callback=self.log.emit,
                progress_callback=self.progress.emit,
            )
            runner.analyze(
                input_folder=self.input_folder,
                project_folder=self.project_folder,
                device=self.device,
                overwrite=True,
            )
            self.completed.emit(str(self.project_folder))
        except Exception:
            self.failed.emit(traceback.format_exc())


class EditorView(QGraphicsView):
    scene_double_clicked = Signal(float, float)

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self._panning = False
        self._last_pan_point = QPoint()

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._last_pan_point
            self._last_pan_point = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is None or isinstance(item, QGraphicsPixmapItem):
            point = self.mapToScene(event.position().toPoint())
            self.scene_double_clicked.emit(point.x(), point.y())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class CellItem(QGraphicsEllipseItem):
    def __init__(
        self,
        cell_id: int,
        class_name: str,
        radius: float,
        x: float,
        y: float,
        edited: bool,
        move_callback,
    ):
        super().__init__(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
        )
        self.cell_id = cell_id
        self.class_name = class_name
        self.move_callback = move_callback
        self.edited = edited

        self.setPos(QPointF(x, y))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(10)
        self.refresh_pen()

    def refresh_pen(self) -> None:
        red, green, blue = CLASS_COLORS_RGB[self.class_name]
        color = QColor(red, green, blue)
        width = 3.2 if self.isSelected() else 1.8
        if self.edited:
            width += 0.8
        self.setPen(QPen(color, width))
        self.setBrush(Qt.BrushStyle.NoBrush)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            QTimer.singleShot(0, self.refresh_pen)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        point = self.pos()
        self.move_callback(
            self.cell_id,
            float(point.x()),
            float(point.y()),
        )


class GeoCombWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoComb")
        self.resize(1500, 900)

        self.database: ProjectDatabase | None = None
        self.current_image_id: int | None = None
        self.current_add_class = "Empty_cell"
        self.cell_items: dict[int, CellItem] = {}
        self.analysis_thread: QThread | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.dirty_image_ids: set[int] = set()
        self.export_timer = QTimer(self)
        self.export_timer.setSingleShot(True)
        self.export_timer.timeout.connect(self.flush_exports)

        self._build_ui()
        self._build_actions()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        project_bar = QHBoxLayout()
        self.project_label = QLabel("No project open")
        self.device_combo = QComboBox()
        self.device_combo.addItems(
            ["auto", "cpu", "cuda"]
        )
        self.open_button = QPushButton("Open project")
        self.analyze_button = QPushButton("Analyze folder")
        self.export_button = QPushButton("Export now")

        self.open_button.clicked.connect(self.choose_project)
        self.analyze_button.clicked.connect(self.choose_analysis)
        self.export_button.clicked.connect(self.flush_exports)

        project_bar.addWidget(self.project_label, 1)
        project_bar.addWidget(QLabel("Device"))
        project_bar.addWidget(self.device_combo)
        project_bar.addWidget(self.open_button)
        project_bar.addWidget(self.analyze_button)
        project_bar.addWidget(self.export_button)
        layout.addLayout(project_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(
            self.image_selected
        )
        splitter.addWidget(self.image_list)

        self.scene = QGraphicsScene(self)
        self.view = EditorView(self.scene)
        self.view.scene_double_clicked.connect(
            self.add_cell_at
        )
        splitter.addWidget(self.view)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        selected_group = QGroupBox("Selected cell")
        selected_layout = QFormLayout(selected_group)
        self.selected_id_label = QLabel("-")
        self.selected_class_label = QLabel("-")
        self.selected_position_label = QLabel("-")
        selected_layout.addRow("Cell ID", self.selected_id_label)
        selected_layout.addRow("Class", self.selected_class_label)
        selected_layout.addRow("Position", self.selected_position_label)
        right_layout.addWidget(selected_group)

        class_group = QGroupBox(
            "Class assignment / new-cell class"
        )
        class_layout = QVBoxLayout(class_group)

        self.class_buttons: dict[str, QPushButton] = {}
        for shortcut, class_name in CLASS_SHORTCUTS.items():
            button = QPushButton(
                f"{shortcut}. {class_name}"
            )
            button.clicked.connect(
                lambda checked=False, name=class_name:
                self.assign_class(name)
            )
            class_layout.addWidget(button)
            self.class_buttons[class_name] = button

        right_layout.addWidget(class_group)

        self.delete_button = QPushButton(
            "Delete selected cell"
        )
        self.delete_button.clicked.connect(
            self.delete_selected
        )
        right_layout.addWidget(self.delete_button)

        counts_group = QGroupBox("Current image counts")
        counts_layout = QVBoxLayout(counts_group)
        self.counts_table = QTableWidget(
            len(CLASS_ORDER) + 1,
            2,
        )
        self.counts_table.setHorizontalHeaderLabels(
            ["Class", "Count"]
        )
        self.counts_table.verticalHeader().setVisible(False)
        counts_layout.addWidget(self.counts_table)
        right_layout.addWidget(counts_group)

        right_layout.addStretch(1)
        splitter.addWidget(right_panel)

        splitter.setSizes([220, 1000, 300])
        layout.addWidget(splitter, 1)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)
        layout.addWidget(self.log_box)

        self.setCentralWidget(central)
        self.scene.selectionChanged.connect(
            self.selection_changed
        )
        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)

        fit_action = QAction("Fit image", self)
        fit_action.triggered.connect(self.fit_image)
        toolbar.addAction(fit_action)

        delete_action = QAction("Delete", self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self.delete_selected)
        self.addAction(delete_action)

        for shortcut, class_name in CLASS_SHORTCUTS.items():
            action = QAction(class_name, self)
            action.setShortcut(str(shortcut))
            action.triggered.connect(
                lambda checked=False, name=class_name:
                self.assign_class(name)
            )
            self.addAction(action)

    def append_log(self, text: str) -> None:
        self.log_box.append(text)
        self.log_box.ensureCursorVisible()

    def choose_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open GeoComb project",
        )
        if folder:
            self.open_project(Path(folder))

    def choose_analysis(self) -> None:
        input_folder = QFileDialog.getExistingDirectory(
            self,
            "Choose image folder",
        )
        if not input_folder:
            return

        project_folder = QFileDialog.getExistingDirectory(
            self,
            "Choose an empty project folder",
        )
        if not project_folder:
            return

        project_path = Path(project_folder)

        if any(project_path.iterdir()):
            response = QMessageBox.question(
                self,
                "Replace project folder?",
                (
                    "The selected project folder is not empty. "
                    "GeoComb will remove its current contents."
                ),
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        self.start_analysis(
            Path(input_folder),
            project_path,
            self.device_combo.currentText(),
        )

    def start_analysis(
        self,
        input_folder: Path,
        project_folder: Path,
        device: str,
    ) -> None:
        self.analyze_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.append_log(
            f"Input: {input_folder}\nProject: {project_folder}"
        )

        thread = QThread(self)
        worker = AnalysisWorker(
            input_folder,
            project_folder,
            device,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.log.connect(self.append_log)
        worker.progress.connect(self.analysis_progress)
        worker.completed.connect(self.analysis_completed)
        worker.failed.connect(self.analysis_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.analysis_thread = thread
        self.analysis_worker = worker
        thread.start()

    def analysis_progress(
        self,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        self.statusBar().showMessage(
            f"{current}/{total}: {stage}"
        )

    def analysis_completed(self, project_folder: str) -> None:
        self.analyze_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.statusBar().showMessage("Analysis complete")
        self.analysis_worker = None
        self.open_project(Path(project_folder))

    def analysis_failed(self, error: str) -> None:
        self.analyze_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.append_log(error)
        QMessageBox.critical(
            self,
            "Analysis failed",
            error,
        )
        self.analysis_worker = None
        self.statusBar().showMessage("Analysis failed")

    def open_project(self, folder: Path) -> None:
        database = ProjectDatabase(folder)

        if not database.database_path.is_file():
            QMessageBox.warning(
                self,
                "Not a GeoComb project",
                f"Missing {database.database_path.name}",
            )
            return

        self.database = database
        self.project_label.setText(str(folder))
        self.image_list.clear()

        for image in database.list_images():
            item = QListWidgetItem(
                f"{image['original_name']} "
                f"({image['active_cells']})"
            )
            item.setData(
                Qt.ItemDataRole.UserRole,
                int(image["id"]),
            )
            self.image_list.addItem(item)

        if self.image_list.count():
            self.image_list.setCurrentRow(0)

    def image_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        if current is None or self.database is None:
            return

        image_id = int(
            current.data(Qt.ItemDataRole.UserRole)
        )
        self.load_image(image_id)

    def load_image(self, image_id: int) -> None:
        assert self.database is not None
        image = self.database.get_image(image_id)
        cells = self.database.cells_for_image(image_id)

        self.current_image_id = image_id
        self.scene.clear()
        self.cell_items.clear()

        pixmap = QPixmap(str(image["crop_image_path"]))
        if pixmap.isNull():
            QMessageBox.critical(
                self,
                "Image error",
                f"Cannot load {image['crop_image_path']}",
            )
            return

        background = self.scene.addPixmap(pixmap)
        background.setZValue(0)
        self.scene.setSceneRect(pixmap.rect())

        for row in cells:
            radius = max(
                5.0,
                float(row["estimated_step_canvas"]) * 0.24,
            )
            item = CellItem(
                cell_id=int(row["id"]),
                class_name=str(row["class_current"]),
                radius=radius,
                x=float(row["x_canvas"]),
                y=float(row["y_canvas"]),
                edited=bool(row["edited"]),
                move_callback=self.cell_moved,
            )
            self.scene.addItem(item)
            self.cell_items[item.cell_id] = item

        self.refresh_counts()
        self.fit_image()

    def fit_image(self) -> None:
        if not self.scene.items():
            return
        self.view.fitInView(
            self.scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def selected_cell_items(self) -> list[CellItem]:
        return [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, CellItem)
        ]

    def selection_changed(self) -> None:
        selected = self.selected_cell_items()

        if not selected:
            self.selected_id_label.setText("-")
            self.selected_class_label.setText("-")
            self.selected_position_label.setText("-")
            return

        item = selected[0]
        point = item.pos()
        self.selected_id_label.setText(str(item.cell_id))
        self.selected_class_label.setText(item.class_name)
        self.selected_position_label.setText(
            f"{point.x():.1f}, {point.y():.1f}"
        )

    def assign_class(self, class_name: str) -> None:
        self.current_add_class = class_name
        selected = self.selected_cell_items()

        if self.database is None:
            return

        for item in selected:
            self.database.set_cell_class(
                item.cell_id,
                class_name,
            )
            item.class_name = class_name
            item.edited = True
            item.refresh_pen()

        if selected:
            self.selection_changed()
            self.data_changed()

        self.statusBar().showMessage(
            f"New cells / selected cells: {class_name}"
        )

    def add_cell_at(self, x: float, y: float) -> None:
        if (
            self.database is None
            or self.current_image_id is None
        ):
            return

        scene_rect = self.scene.sceneRect()
        if not scene_rect.contains(QPointF(x, y)):
            return

        cell_id = self.database.add_cell(
            self.current_image_id,
            x,
            y,
            self.current_add_class,
        )
        rows = self.database.cells_for_image(
            self.current_image_id
        )
        row = next(
            value
            for value in rows
            if int(value["id"]) == cell_id
        )
        radius = max(
            5.0,
            float(row["estimated_step_canvas"]) * 0.24,
        )
        item = CellItem(
            cell_id=cell_id,
            class_name=self.current_add_class,
            radius=radius,
            x=x,
            y=y,
            edited=True,
            move_callback=self.cell_moved,
        )
        self.scene.addItem(item)
        self.cell_items[cell_id] = item
        item.setSelected(True)
        self.data_changed()

    def cell_moved(
        self,
        cell_id: int,
        x: float,
        y: float,
    ) -> None:
        if self.database is None:
            return

        self.database.move_cell(
            cell_id,
            x,
            y,
        )
        item = self.cell_items.get(cell_id)
        if item is not None:
            item.edited = True
            item.refresh_pen()
        self.selection_changed()
        self.data_changed()

    def delete_selected(self) -> None:
        if self.database is None:
            return

        selected = self.selected_cell_items()
        if not selected:
            return

        for item in selected:
            self.database.set_deleted(
                item.cell_id,
                True,
            )
            self.scene.removeItem(item)
            self.cell_items.pop(item.cell_id, None)

        self.data_changed()

    def refresh_counts(self) -> None:
        if (
            self.database is None
            or self.current_image_id is None
        ):
            return

        counts = self.database.class_counts(
            self.current_image_id
        )

        for row_index, class_name in enumerate(CLASS_ORDER):
            self.counts_table.setItem(
                row_index,
                0,
                QTableWidgetItem(class_name),
            )
            self.counts_table.setItem(
                row_index,
                1,
                QTableWidgetItem(str(counts[class_name])),
            )

        total_row = len(CLASS_ORDER)
        self.counts_table.setItem(
            total_row,
            0,
            QTableWidgetItem("Total"),
        )
        self.counts_table.setItem(
            total_row,
            1,
            QTableWidgetItem(str(sum(counts.values()))),
        )
        self.counts_table.resizeColumnsToContents()

        total = sum(counts.values())
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if int(item.data(Qt.ItemDataRole.UserRole)) == self.current_image_id:
                image = self.database.get_image(self.current_image_id)
                item.setText(f"{image['original_name']} ({total})")
                break

    def data_changed(self) -> None:
        if self.current_image_id is not None:
            self.dirty_image_ids.add(self.current_image_id)
        self.refresh_counts()
        self.export_timer.start(1200)

    def flush_exports(self) -> None:
        if self.database is None:
            return

        try:
            dirty = set(self.dirty_image_ids)
            if not dirty and self.current_image_id is not None:
                dirty.add(self.current_image_id)

            for image_id in sorted(dirty):
                export_image_overlays(
                    self.database,
                    image_id,
                )
            self.dirty_image_ids.difference_update(dirty)

            self.database.export_cells_csv(
                self.database.project_folder
                / "results"
                / "cells"
            )
            path, ok, message = export_excel(
                self.database
            )
            self.statusBar().showMessage(
                f"{message}: {path}"
            )
        except Exception:
            error = traceback.format_exc()
            self.append_log(error)
            self.statusBar().showMessage(
                "Automatic export failed; see log."
            )


def run_gui() -> int:
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("GeoComb")
    window = GeoCombWindow()
    window.show()
    return application.exec()
