#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

IDS = ["1", "9", "21", "26", "42", "51"]

CLASSES8 = [
    "Egg", "Larvae", "Capped_pupae", "Pollen",
    "Capped_honey", "Uncapped_honey", "Empty_cell", "Unbuilt_cell",
]
CLASSES7 = ["Egg", "Larvae", "Capped_brood", "Pollen", "Honey", "Nectar", "Other"]

MAP8TO7 = {
    "Egg": "Egg", "Larvae": "Larvae", "Capped_pupae": "Capped_brood",
    "Pollen": "Pollen", "Capped_honey": "Honey",
    "Uncapped_honey": "Nectar", "Empty_cell": "Other", "Unbuilt_cell": "Other",
}
DEEPBEE_ID = {
    0: "Capped_brood", 1: "Egg", 2: "Honey", 3: "Larvae",
    4: "Nectar", 5: "Other", 6: "Pollen",
}
P_ALIASES = {
    "egg": "Egg", "larva": "Larvae", "larvae": "Larvae",
    "cappedpupa": "Capped_pupae", "cappedpupae": "Capped_pupae",
    "cappedbrood": "Capped_pupae", "pollen": "Pollen",
    "cappedhoney": "Capped_honey", "honey": "Capped_honey",
    "uncappedhoney": "Uncapped_honey", "nectar": "Uncapped_honey",
    "empty": "Empty_cell", "emptycell": "Empty_cell",
    "unbuilt": "Unbuilt_cell", "unbuiltcell": "Unbuilt_cell",
}
D_ALIASES = {
    "capped": "Capped_brood", "cappedbrood": "Capped_brood",
    "egg": "Egg", "eggs": "Egg", "honey": "Honey",
    "larva": "Larvae", "larvae": "Larvae", "larves": "Larvae",
    "nectar": "Nectar", "other": "Other", "pollen": "Pollen",
}


def token(value: Any) -> str:
    return "".join(c for c in str(value).strip().lower() if c.isalnum())


def find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {token(c): c for c in df.columns}
    for name in names:
        if token(name) in lookup:
            return lookup[token(name)]
    return None


def norm8(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Missing 8-class label.")
    try:
        n = float(value)
        if np.isfinite(n) and n.is_integer() and 0 <= int(n) < len(CLASSES8):
            return CLASSES8[int(n)]
    except (TypeError, ValueError):
        pass
    key = token(value)
    if key not in P_ALIASES:
        raise ValueError(f"Unknown 8-class label: {value!r}")
    return P_ALIASES[key]


def norm7(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Missing DeepBee class.")
    try:
        n = float(value)
        if np.isfinite(n) and n.is_integer() and int(n) in DEEPBEE_ID:
            return DEEPBEE_ID[int(n)]
    except (TypeError, ValueError):
        pass
    key = token(value)
    if key not in D_ALIASES:
        raise ValueError(f"Unknown DeepBee label: {value!r}")
    return D_ALIASES[key]


def load_ref(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    x = find_col(df, ["x", "center_x", "cx"])
    y = find_col(df, ["y", "center_y", "cy"])
    c = find_col(df, ["class_name", "class8", "class", "label"])
    if None in (x, y, c):
        raise ValueError(f"Reference columns not recognized in {path}: {list(df.columns)}")
    out = pd.DataFrame({
        "x": pd.to_numeric(df[x], errors="coerce"),
        "y": pd.to_numeric(df[y], errors="coerce"),
        "class8": df[c].map(norm8),
    }).dropna(subset=["x", "y"])
    out["class7"] = out["class8"].map(MAP8TO7)
    return out.reset_index(drop=True)


def load_geocomb(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    x = find_col(df, ["x", "center_x", "cx", "x_center", "pred_x", "cell_x"])
    y = find_col(df, ["y", "center_y", "cy", "y_center", "pred_y", "cell_y"])
    c = find_col(df, [
        "class_name", "pred_class8", "pred_class", "final_class",
        "class8", "class", "label", "prediction",
    ])
    if None in (x, y, c):
        raise ValueError(f"GeoComb columns not recognized in {path}: {list(df.columns)}")
    out = pd.DataFrame({
        "x": pd.to_numeric(df[x], errors="coerce"),
        "y": pd.to_numeric(df[y], errors="coerce"),
        "class8": df[c].map(norm8),
    }).dropna(subset=["x", "y"])
    out["class7"] = out["class8"].map(MAP8TO7)
    return out.reset_index(drop=True)


def load_deepbee(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.load(path, allow_pickle=True)
    if not (isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == (3,)):
        raise ValueError(f"Unexpected DeepBee npy format: {path}, shape={getattr(raw,'shape',None)}")
    pts = np.asarray(raw[2], dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 5:
        raise ValueError(f"Unexpected DeepBee points: {path}, shape={pts.shape}")
    if pts.shape[1] > 6:
        pts = pts[pts[:, 6] != 0]
    class_col = 4 if pts.shape[1] > 4 else 3
    return pd.DataFrame({
        "x": pts[:, 0],
        "y": pts[:, 1],
        "class7": [norm7(v) for v in pts[:, class_col]],
    }).reset_index(drop=True)


def estimate_pitch(xy: np.ndarray, fallback: float = 60.0) -> float:
    if len(xy) < 10:
        return fallback
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 2)]
    if len(nn) < 10:
        return fallback
    lo, hi = np.quantile(nn, [0.1, 0.9])
    nn = nn[(nn >= lo) & (nn <= hi)]
    pitch = float(np.median(nn))
    return pitch if 5 <= pitch <= 300 else fallback


def get_pitch(folder: Path, ref: pd.DataFrame) -> float:
    manifest = folder / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if "pitch" in data:
            return float(data["pitch"])
    return estimate_pitch(ref[["x", "y"]].to_numpy(float))


def match(ref_xy: np.ndarray, pred_xy: np.ndarray, radius: float):
    if len(ref_xy) == 0 or len(pred_xy) == 0:
        return []
    tree = cKDTree(pred_xy)
    pairs = []
    for i, point in enumerate(ref_xy):
        for j in tree.query_ball_point(point, radius):
            pairs.append((float(np.linalg.norm(point - pred_xy[j])), i, j))
    pairs.sort()
    used_r, used_p, out = set(), set(), []
    for dist, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        out.append((i, j, dist))
    return out


def ratio(a: float, b: float) -> float:
    return float(a / b) if b else math.nan


def det_metrics(n_ref: int, n_pred: int, tp: int) -> dict:
    fp = n_pred - tp
    fn = n_ref - tp
    p = ratio(tp, tp + fp)
    r = ratio(tp, tp + fn)
    f1 = ratio(2 * p * r, p + r)
    return {
        "reference_count": n_ref,
        "prediction_count": n_pred,
        "TP": tp, "FP": fp, "FN": fn,
        "detection_precision": p,
        "detection_recall": r,
        "detection_F1": f1,
    }


def class_metrics(y_true, y_pred, labels, ref_total):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    table = pd.DataFrame({
        "class": labels, "precision": p, "recall": r,
        "F1": f1, "support": support.astype(int),
    })
    supported = support > 0
    accuracy = float(accuracy_score(y_true, y_pred))
    correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
    overall = {
        "matched_cells": len(y_true),
        "matched_class_accuracy": accuracy,
        "macro_F1_supported_classes": float(np.mean(f1[supported])) if supported.any() else math.nan,
        "macro_F1_all_labels": float(np.mean(f1)),
        "weighted_F1": float(np.average(f1, weights=support)) if support.sum() else math.nan,
        "correctly_classified_matches": correct,
        "end_to_end_correct_fraction": ratio(correct, ref_total),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return overall, table, cm


def class_count_error(ref_labels: pd.Series, pred_labels: pd.Series, labels: list[str]) -> float:
    values = []
    for label in labels:
        nr = int((ref_labels == label).sum())
        npred = int((pred_labels == label).sum())
        values.append(abs(npred - nr) / max(nr, 1))
    return float(np.mean(values))


def save_cm(cm, labels, name, outdir):
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name = "Reference"
    df.columns.name = "Prediction"
    df.to_csv(outdir / f"confusion_matrix_{name}.csv")

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), max(6, len(labels) * 0.8)))
    im = ax.imshow(cm, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(name.replace("_", " "))
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Reference")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_matrix_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_classification(name, method, scheme, labels, refs, preds, matches):
    class_col = "class8" if scheme == "8class_separated" else "class7"
    y_true, y_pred = [], []
    total_ref = 0
    per_image = []

    for image_id in IDS:
        ref = refs[image_id]
        pred = preds[image_id]
        pairs = matches[image_id]
        yt = [str(ref.iloc[i][class_col]) for i, _, _ in pairs]
        yp = [str(pred.iloc[j][class_col]) for _, j, _ in pairs]
        y_true.extend(yt)
        y_pred.extend(yp)
        total_ref += len(ref)
        correct = int(np.sum(np.asarray(yt) == np.asarray(yp)))
        per_image.append({
            "image_id": image_id,
            "evaluation": name,
            "matched_cells": len(pairs),
            "matched_class_accuracy": float(accuracy_score(yt, yp)) if yt else math.nan,
            "end_to_end_correct_fraction": ratio(correct, len(ref)),
            "normalized_class_count_error": class_count_error(
                ref[class_col], pred[class_col], labels
            ),
        })

    overall, class_table, cm = class_metrics(y_true, y_pred, labels, total_ref)
    per_image_df = pd.DataFrame(per_image)
    overall["method"] = method
    overall["class_scheme"] = scheme
    overall["evaluation"] = name
    overall["mean_normalized_class_count_error"] = float(
        per_image_df["normalized_class_count_error"].mean()
    )
    class_table.insert(0, "evaluation", name)
    return overall, class_table, cm, per_image_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--geocomb-root", type=Path, required=True)
    parser.add_argument("--deepbee-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--match-radius-fraction", type=float, default=0.50)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    refs, geocomb, deepbee = {}, {}, {}
    pmatches, dmatches = {}, {}
    det_rows = []

    for image_id in IDS:
        ref_dir = args.reference_root / image_id
        ref_path = ref_dir / "manual_reference_8class.csv"
        p_path = args.geocomb_root / image_id / "cell_predictions_3stage.csv"
        d_path = args.deepbee_root / f"{image_id}.npy"
        for path in (ref_path, p_path, d_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        ref = load_ref(ref_path)
        p = load_geocomb(p_path)
        d = load_deepbee(d_path)
        pitch = get_pitch(ref_dir, ref)
        radius = pitch * args.match_radius_fraction

        mp = match(ref[["x", "y"]].to_numpy(float), p[["x", "y"]].to_numpy(float), radius)
        md = match(ref[["x", "y"]].to_numpy(float), d[["x", "y"]].to_numpy(float), radius)

        refs[image_id], geocomb[image_id], deepbee[image_id] = ref, p, d
        pmatches[image_id], dmatches[image_id] = mp, md

        for method, pred, pairs in [("GeoComb", p, mp), ("DeepBee", d, md)]:
            row = det_metrics(len(ref), len(pred), len(pairs))
            row.update({
                "image_id": image_id,
                "method": method,
                "pitch_px": pitch,
                "match_threshold_px": radius,
                "relative_total_count_error": ratio(abs(len(pred) - len(ref)), len(ref)),
                "mean_matched_distance_px": float(np.mean([x[2] for x in pairs])) if pairs else math.nan,
            })
            det_rows.append(row)

        print(
            f"{image_id}: reference={len(ref)}, pitch={pitch:.2f}, "
            f"GeoComb TP={len(mp)}, DeepBee TP={len(md)}"
        )

    det = pd.DataFrame(det_rows)
    det.to_csv(args.output_root / "per_image_detection.csv", index=False)

    det_overall = []
    for method in ("GeoComb", "DeepBee"):
        sub = det[det["method"] == method]
        tp = int(sub["TP"].sum())
        row = det_metrics(
            int(sub["reference_count"].sum()),
            int(sub["prediction_count"].sum()),
            tp,
        )
        row.update({
            "method": method,
            "mean_relative_total_count_error": float(sub["relative_total_count_error"].mean()),
            "mean_matched_distance_px": float(sub["mean_matched_distance_px"].mean()),
        })
        det_overall.append(row)
    det_overall = pd.DataFrame(det_overall)
    det_overall.to_csv(args.output_root / "overall_detection.csv", index=False)

    evaluations = [
        ("geocomb_7class_merged", "GeoComb", "7class_merged", CLASSES7, geocomb, pmatches),
        ("deepbee_7class_native", "DeepBee", "7class_native", CLASSES7, deepbee, dmatches),
        ("geocomb_8class_separated", "GeoComb", "8class_separated", CLASSES8, geocomb, pmatches),
    ]

    overall_rows = []

    for name, method, scheme, labels, preds, matches in evaluations:
        overall, class_table, cm, per_image = evaluate_classification(
            name, method, scheme, labels, refs, preds, matches
        )
        det_row = det_overall[det_overall["method"] == method].iloc[0].to_dict()
        overall.update({k: v for k, v in det_row.items() if k != "method"})
        overall_rows.append(overall)
        class_table.to_csv(args.output_root / f"class_metrics_{name}.csv", index=False)
        per_image.to_csv(args.output_root / f"per_image_classification_{name}.csv", index=False)
        save_cm(cm, labels, name, args.output_root)

    overall = pd.DataFrame(overall_rows)
    order = [
        "evaluation", "method", "class_scheme",
        "reference_count", "prediction_count", "TP", "FP", "FN",
        "detection_precision", "detection_recall", "detection_F1",
        "mean_relative_total_count_error",
        "matched_cells", "matched_class_accuracy",
        "macro_F1_supported_classes", "macro_F1_all_labels", "weighted_F1",
        "correctly_classified_matches", "end_to_end_correct_fraction",
        "mean_normalized_class_count_error", "mean_matched_distance_px",
    ]
    overall = overall[[c for c in order if c in overall.columns]]
    overall.to_csv(args.output_root / "overall_comparison.csv", index=False)

    plot_cols = [
        "detection_precision", "detection_recall", "detection_F1",
        "matched_class_accuracy", "macro_F1_supported_classes",
        "weighted_F1", "end_to_end_correct_fraction",
    ]
    ax = overall.set_index("evaluation")[plot_cols].T.plot(kind="bar", figsize=(11, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("In-house evaluation summary")
    ax.tick_params(axis="x", rotation=35)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(args.output_root / "overall_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "IN-HOUSE FINAL EVALUATION",
        "=" * 78,
        "Reference: DeepBee-seeded and manually corrected.",
        f"Matching threshold: {args.match_radius_fraction:.2f} x image-specific pitch.",
        "",
        "SEVEN-CLASS DIRECT COMPARISON",
        "-" * 78,
    ]
    for name in ("geocomb_7class_merged", "deepbee_7class_native"):
        row = overall[overall["evaluation"] == name].iloc[0]
        lines += [
            name,
            f"  Detection P/R/F1: {row.detection_precision:.6f} / "
            f"{row.detection_recall:.6f} / {row.detection_F1:.6f}",
            f"  Mean total count error: {row.mean_relative_total_count_error:.6f}",
            f"  Matched accuracy: {row.matched_class_accuracy:.6f}",
            f"  Macro F1 (supported): {row.macro_F1_supported_classes:.6f}",
            f"  Weighted F1: {row.weighted_F1:.6f}",
            f"  End-to-end correct fraction: {row.end_to_end_correct_fraction:.6f}",
            f"  Mean normalized class-count error: "
            f"{row.mean_normalized_class_count_error:.6f}",
            "",
        ]

    row = overall[overall["evaluation"] == "geocomb_8class_separated"].iloc[0]
    lines += [
        "GEOCOMB EIGHT-CLASS SEPARATED RESULT",
        "-" * 78,
        f"Detection P/R/F1: {row.detection_precision:.6f} / "
        f"{row.detection_recall:.6f} / {row.detection_F1:.6f}",
        f"Matched accuracy: {row.matched_class_accuracy:.6f}",
        f"Macro F1 (supported): {row.macro_F1_supported_classes:.6f}",
        f"Weighted F1: {row.weighted_F1:.6f}",
        f"End-to-end correct fraction: {row.end_to_end_correct_fraction:.6f}",
        f"Mean normalized class-count error: "
        f"{row.mean_normalized_class_count_error:.6f}",
        "",
        "DeepBee has only one Other class, so no eight-class DeepBee score is reported.",
        "",
        "Potential bias: the manual reference was initialized from DeepBee predictions.",
    ]
    summary = args.output_root / "evaluation_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(overall.to_string(index=False))
    print()
    print(f"Saved to: {args.output_root}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
