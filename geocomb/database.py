from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image

from .constants import CLASS_ORDER


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    safe_stem TEXT NOT NULL UNIQUE,
    safe_name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    copied_path TEXT NOT NULL,
    crop_image_path TEXT NOT NULL,
    original_width INTEGER NOT NULL,
    original_height INTEGER NOT NULL,
    canvas_width INTEGER NOT NULL,
    canvas_height INTEGER NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    scale_5000 REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'analyzed',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    source_row INTEGER,
    x_canvas REAL NOT NULL,
    y_canvas REAL NOT NULL,
    x_original REAL NOT NULL,
    y_original REAL NOT NULL,
    estimated_step_canvas REAL NOT NULL,
    class_initial TEXT NOT NULL,
    class_current TEXT NOT NULL,
    confidence REAL,
    edited INTEGER NOT NULL DEFAULT 0,
    added INTEGER NOT NULL DEFAULT 0,
    deleted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cells_image
ON cells(image_id);

CREATE INDEX IF NOT EXISTS idx_cells_active
ON cells(image_id, deleted);

CREATE TABLE IF NOT EXISTS edit_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_id INTEGER NOT NULL,
    cell_id INTEGER,
    action TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);
"""


class ProjectDatabase:
    def __init__(self, project_folder: Path):
        self.project_folder = Path(project_folder).resolve()
        self.database_path = self.project_folder / "project.geocomb"

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.project_folder.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def set_meta(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO project_meta(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, encoded),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM project_meta WHERE key=?",
                (key,),
            ).fetchone()
        return default if row is None else json.loads(row["value"])

    def import_pipeline_results(
        self,
        input_manifest: list[dict],
        crop_summary_path: Path,
        prediction_root: Path,
        crop_image_root: Path,
        replace_existing: bool = True,
    ) -> None:
        self.initialize()

        crop_summary = pd.read_csv(crop_summary_path)
        crop_summary["image"] = crop_summary["image"].astype(str)

        manifest_by_stem = {
            str(row["safe_stem"]): row
            for row in input_manifest
        }

        with self.connect() as connection:
            if replace_existing:
                connection.execute("DELETE FROM edit_history")
                connection.execute("DELETE FROM cells")
                connection.execute("DELETE FROM images")

            for safe_stem, manifest in manifest_by_stem.items():
                crop_rows = crop_summary.loc[
                    crop_summary["image"].eq(safe_stem)
                ]

                if crop_rows.empty:
                    raise RuntimeError(
                        f"Crop metadata missing for {safe_stem}"
                    )

                crop = crop_rows.iloc[0]
                prediction_csv = (
                    prediction_root
                    / "test"
                    / safe_stem
                    / "cell_predictions_3stage.csv"
                )
                crop_image = crop_image_root / "test" / f"{safe_stem}.png"

                if not prediction_csv.is_file():
                    raise FileNotFoundError(prediction_csv)

                if not crop_image.is_file():
                    raise FileNotFoundError(crop_image)

                with Image.open(manifest["copied_path"]) as image:
                    original_width, original_height = image.size

                with Image.open(crop_image) as image:
                    canvas_width, canvas_height = image.size

                now = utc_now()

                cursor = connection.execute(
                    """
                    INSERT INTO images(
                        safe_stem, safe_name, original_name, source_path,
                        copied_path, crop_image_path,
                        original_width, original_height,
                        canvas_width, canvas_height,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        scale_5000, status, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        safe_stem,
                        manifest["safe_name"],
                        manifest["original_name"],
                        manifest["source_path"],
                        manifest["copied_path"],
                        str(crop_image.resolve()),
                        int(original_width),
                        int(original_height),
                        int(canvas_width),
                        int(canvas_height),
                        float(crop["bbox_x1"]),
                        float(crop["bbox_y1"]),
                        float(crop["bbox_x2"]),
                        float(crop["bbox_y2"]),
                        float(crop["scale_5000"]),
                        "analyzed",
                        now,
                        now,
                    ),
                )
                image_id = int(cursor.lastrowid)

                predictions = pd.read_csv(prediction_csv)

                for source_row, row in predictions.iterrows():
                    class_name = str(row.get("final_class", ""))
                    if class_name not in CLASS_ORDER:
                        raise ValueError(
                            f"Unknown final class {class_name!r} "
                            f"in {prediction_csv}"
                        )

                    x_canvas = float(row["x"])
                    y_canvas = float(row["y"])
                    scale = float(crop["scale_5000"])
                    x_original = (
                        x_canvas / scale + float(crop["bbox_x1"])
                    )
                    y_original = (
                        y_canvas / scale + float(crop["bbox_y1"])
                    )
                    step = float(
                        row.get(
                            "estimated_step",
                            np.nan,
                        )
                    )
                    if not np.isfinite(step) or step <= 0:
                        step = 60.0

                    confidence = row.get("final_conf", np.nan)
                    confidence_value = (
                        None
                        if pd.isna(confidence)
                        else float(confidence)
                    )

                    connection.execute(
                        """
                        INSERT INTO cells(
                            image_id, source_row,
                            x_canvas, y_canvas,
                            x_original, y_original,
                            estimated_step_canvas,
                            class_initial, class_current,
                            confidence, edited, added, deleted,
                            created_at, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?)
                        """,
                        (
                            image_id,
                            int(source_row),
                            x_canvas,
                            y_canvas,
                            x_original,
                            y_original,
                            step,
                            class_name,
                            class_name,
                            confidence_value,
                            now,
                            now,
                        ),
                    )

    def list_images(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    i.*,
                    SUM(CASE WHEN c.deleted=0 THEN 1 ELSE 0 END) AS active_cells
                FROM images i
                LEFT JOIN cells c ON c.image_id=i.id
                GROUP BY i.id
                ORDER BY i.id
                """
            ).fetchall()
        return rows

    def get_image(self, image_id: int) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM images WHERE id=?",
                (image_id,),
            ).fetchone()
        if row is None:
            raise KeyError(image_id)
        return row

    def cells_for_image(
        self,
        image_id: int,
        include_deleted: bool = False,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM cells WHERE image_id=?"
        parameters: list[Any] = [image_id]

        if not include_deleted:
            query += " AND deleted=0"

        query += " ORDER BY id"

        with self.connect() as connection:
            return connection.execute(
                query,
                parameters,
            ).fetchall()

    def class_counts(self, image_id: int) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_ORDER}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT class_current, COUNT(*) AS n
                FROM cells
                WHERE image_id=? AND deleted=0
                GROUP BY class_current
                """,
                (image_id,),
            ).fetchall()

        for row in rows:
            counts[str(row["class_current"])] = int(row["n"])
        return counts

    def _original_coordinates(
        self,
        image: sqlite3.Row,
        x_canvas: float,
        y_canvas: float,
    ) -> tuple[float, float]:
        scale = float(image["scale_5000"])
        return (
            x_canvas / scale + float(image["bbox_x1"]),
            y_canvas / scale + float(image["bbox_y1"]),
        )

    def move_cell(
        self,
        cell_id: int,
        x_canvas: float,
        y_canvas: float,
    ) -> None:
        with self.connect() as connection:
            cell = connection.execute(
                "SELECT * FROM cells WHERE id=?",
                (cell_id,),
            ).fetchone()
            if cell is None:
                raise KeyError(cell_id)

            if (
                abs(float(cell["x_canvas"]) - float(x_canvas)) < 1e-6
                and abs(float(cell["y_canvas"]) - float(y_canvas)) < 1e-6
            ):
                return

            image = connection.execute(
                "SELECT * FROM images WHERE id=?",
                (cell["image_id"],),
            ).fetchone()
            x_original, y_original = self._original_coordinates(
                image,
                x_canvas,
                y_canvas,
            )

            old_value = {
                "x_canvas": cell["x_canvas"],
                "y_canvas": cell["y_canvas"],
                "x_original": cell["x_original"],
                "y_original": cell["y_original"],
            }
            new_value = {
                "x_canvas": float(x_canvas),
                "y_canvas": float(y_canvas),
                "x_original": float(x_original),
                "y_original": float(y_original),
            }
            now = utc_now()

            connection.execute(
                """
                UPDATE cells
                SET x_canvas=?, y_canvas=?,
                    x_original=?, y_original=?,
                    edited=1, updated_at=?
                WHERE id=?
                """,
                (
                    float(x_canvas),
                    float(y_canvas),
                    float(x_original),
                    float(y_original),
                    now,
                    cell_id,
                ),
            )
            self._history(
                connection,
                int(cell["image_id"]),
                cell_id,
                "move",
                old_value,
                new_value,
            )

    def set_cell_class(
        self,
        cell_id: int,
        class_name: str,
    ) -> None:
        if class_name not in CLASS_ORDER:
            raise ValueError(class_name)

        with self.connect() as connection:
            cell = connection.execute(
                "SELECT * FROM cells WHERE id=?",
                (cell_id,),
            ).fetchone()
            if cell is None:
                raise KeyError(cell_id)

            if cell["class_current"] == class_name:
                return

            now = utc_now()
            connection.execute(
                """
                UPDATE cells
                SET class_current=?, edited=1, updated_at=?
                WHERE id=?
                """,
                (class_name, now, cell_id),
            )
            self._history(
                connection,
                int(cell["image_id"]),
                cell_id,
                "class_change",
                {"class": cell["class_current"]},
                {"class": class_name},
            )

    def add_cell(
        self,
        image_id: int,
        x_canvas: float,
        y_canvas: float,
        class_name: str,
    ) -> int:
        if class_name not in CLASS_ORDER:
            raise ValueError(class_name)

        with self.connect() as connection:
            image = connection.execute(
                "SELECT * FROM images WHERE id=?",
                (image_id,),
            ).fetchone()
            if image is None:
                raise KeyError(image_id)

            step_row = connection.execute(
                """
                SELECT AVG(estimated_step_canvas) AS mean_step
                FROM cells
                WHERE image_id=? AND deleted=0
                """,
                (image_id,),
            ).fetchone()
            step = float(step_row["mean_step"] or 60.0)
            x_original, y_original = self._original_coordinates(
                image,
                x_canvas,
                y_canvas,
            )
            now = utc_now()

            cursor = connection.execute(
                """
                INSERT INTO cells(
                    image_id, source_row,
                    x_canvas, y_canvas,
                    x_original, y_original,
                    estimated_step_canvas,
                    class_initial, class_current,
                    confidence, edited, added, deleted,
                    created_at, updated_at
                )
                VALUES(?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 1, 1, 0, ?, ?)
                """,
                (
                    image_id,
                    float(x_canvas),
                    float(y_canvas),
                    float(x_original),
                    float(y_original),
                    step,
                    class_name,
                    class_name,
                    now,
                    now,
                ),
            )
            cell_id = int(cursor.lastrowid)
            self._history(
                connection,
                image_id,
                cell_id,
                "add",
                None,
                {
                    "x_canvas": x_canvas,
                    "y_canvas": y_canvas,
                    "class": class_name,
                },
            )
            return cell_id

    def set_deleted(
        self,
        cell_id: int,
        deleted: bool = True,
    ) -> None:
        with self.connect() as connection:
            cell = connection.execute(
                "SELECT * FROM cells WHERE id=?",
                (cell_id,),
            ).fetchone()
            if cell is None:
                raise KeyError(cell_id)

            new_value = 1 if deleted else 0
            if int(cell["deleted"]) == new_value:
                return

            now = utc_now()
            connection.execute(
                """
                UPDATE cells
                SET deleted=?, edited=1, updated_at=?
                WHERE id=?
                """,
                (new_value, now, cell_id),
            )
            self._history(
                connection,
                int(cell["image_id"]),
                cell_id,
                "delete" if deleted else "restore",
                {"deleted": int(cell["deleted"])},
                {"deleted": new_value},
            )

    def _history(
        self,
        connection: sqlite3.Connection,
        image_id: int,
        cell_id: int | None,
        action: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        connection.execute(
            """
            INSERT INTO edit_history(
                timestamp, image_id, cell_id,
                action, old_value, new_value
            )
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                image_id,
                cell_id,
                action,
                None
                if old_value is None
                else json.dumps(old_value, ensure_ascii=False),
                None
                if new_value is None
                else json.dumps(new_value, ensure_ascii=False),
            ),
        )

    def all_cells(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.*,
                    i.original_name,
                    i.safe_stem
                FROM cells c
                JOIN images i ON i.id=c.image_id
                ORDER BY i.id, c.id
                """
            ).fetchall()

    def edit_history(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    h.*,
                    i.original_name
                FROM edit_history h
                JOIN images i ON i.id=h.image_id
                ORDER BY h.id
                """
            ).fetchall()

    def export_cells_csv(self, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)

        for image in self.list_images():
            rows = self.cells_for_image(
                int(image["id"]),
                include_deleted=True,
            )
            path = output_folder / f"{image['safe_stem']}_cells.csv"

            fields = [
                "cell_id",
                "x_canvas",
                "y_canvas",
                "x_original",
                "y_original",
                "estimated_step_canvas",
                "class_initial",
                "class_current",
                "confidence",
                "edited",
                "added",
                "deleted",
            ]

            with path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "cell_id": row["id"],
                            "x_canvas": row["x_canvas"],
                            "y_canvas": row["y_canvas"],
                            "x_original": row["x_original"],
                            "y_original": row["y_original"],
                            "estimated_step_canvas": row[
                                "estimated_step_canvas"
                            ],
                            "class_initial": row["class_initial"],
                            "class_current": row["class_current"],
                            "confidence": row["confidence"],
                            "edited": row["edited"],
                            "added": row["added"],
                            "deleted": row["deleted"],
                        }
                    )
