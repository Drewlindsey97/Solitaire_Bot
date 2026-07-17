from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardLayout:
    reference_width: int = 720
    reference_height: int = 1600
    tableau_x: tuple[int, ...] = (10, 111, 212, 313, 414, 512, 611)
    tableau_y_top: int = 507
    column_width: int = 95
    revealed_card_step: int = 50
    hidden_card_step: int = 30
    free_cell_x: tuple[int, ...] = (10, 110, 210, 310)
    foundation_x: tuple[int, ...] = (472,)
    slot_y: int = 303
    slot_width: int = 95
    slot_height: int = 90
    rank_width: int = 45
    rank_height: int = 45
    suit_x_offset: int = 5
    suit_y_offset: int = 42
    suit_width: int = 35
    suit_height: int = 22
    last_card_crop_width: int = 95
    last_card_crop_height: int = 90
    full_card_height: int = 135
    max_tableau_cards: int = 20
    min_rank_score: float = 0.55

    @classmethod
    def from_json(cls, path: str | Path) -> "BoardLayout":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("tableau_x", "free_cell_x", "foundation_x"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def to_json_data(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoardTransform:
    original_width: int
    original_height: int
    content_x: float
    content_y: float
    content_width: float
    content_height: float
    reference_width: int
    reference_height: int

    @property
    def scale_x(self) -> float:
        return self.reference_width / self.content_width

    @property
    def scale_y(self) -> float:
        return self.reference_height / self.content_height

    def original_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.content_x) * self.scale_x, (y - self.content_y) * self.scale_y)

    def normalized_to_original(self, x: float, y: float) -> tuple[float, float]:
        return (x / self.scale_x + self.content_x, y / self.scale_y + self.content_y)

    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        x1 = max(0, int(round(self.content_x)))
        y1 = max(0, int(round(self.content_y)))
        x2 = min(self.original_width, int(round(self.content_x + self.content_width)))
        y2 = min(self.original_height, int(round(self.content_y + self.content_height)))
        cropped = img[y1:y2, x1:x2]
        if cropped.shape[1] == self.reference_width and cropped.shape[0] == self.reference_height:
            return cropped.copy()
        return cv2.resize(cropped, (self.reference_width, self.reference_height), interpolation=cv2.INTER_AREA)

    def to_json_data(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale_x"] = self.scale_x
        data["scale_y"] = self.scale_y
        return data


@dataclass(frozen=True)
class ObservedColumn:
    hidden_count: int
    visible_cards: tuple[dict[str, Any], ...]

    def to_json_data(self) -> dict[str, Any]:
        return {
            "hidden_count": self.hidden_count,
            "visible_cards": list(self.visible_cards),
        }


@dataclass(frozen=True)
class BoardObservation:
    columns: tuple[ObservedColumn, ...]
    free_cells: tuple[dict[str, Any] | None, ...]
    foundation: tuple[dict[str, Any] | None, ...]

    def to_json_data(self) -> dict[str, Any]:
        return {
            "columns": [column.to_json_data() for column in self.columns],
            "free_cells": list(self.free_cells),
            "foundation": list(self.foundation),
        }


REFERENCE_LAYOUT = BoardLayout()

# Backward-compatible exports for existing helpers.
STEP = REFERENCE_LAYOUT.revealed_card_step
RANK_W, RANK_H = REFERENCE_LAYOUT.rank_width, REFERENCE_LAYOUT.rank_height
SUIT_X_OFF, SUIT_Y_OFF = REFERENCE_LAYOUT.suit_x_offset, REFERENCE_LAYOUT.suit_y_offset
SUIT_W, SUIT_H = REFERENCE_LAYOUT.suit_width, REFERENCE_LAYOUT.suit_height
PAD = 15
HIDDEN_CARD_H = REFERENCE_LAYOUT.hidden_card_step
TOP_RESIDUAL_TOLERANCE = 5
TABLEAU_X = list(REFERENCE_LAYOUT.tableau_x)
TABLEAU_Y_TOP = REFERENCE_LAYOUT.tableau_y_top
COL_WIDTH = REFERENCE_LAYOUT.column_width
FREE_CELL_X = list(REFERENCE_LAYOUT.free_cell_x)
FOUNDATION_X = list(REFERENCE_LAYOUT.foundation_x)
SLOT_Y = REFERENCE_LAYOUT.slot_y
SLOT_W, SLOT_H = REFERENCE_LAYOUT.slot_width, REFERENCE_LAYOUT.slot_height


def _template_dir(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def load_templates(folder):
    t = {}
    for path in glob.glob(f"{folder}/*.png"):
        name = os.path.splitext(os.path.basename(path))[0]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t[name] = binary
    return t


TEMPLATES = load_templates(_template_dir("templates"))
TEMPLATES_LAST = load_templates(_template_dir("templates_last"))


def classify_suit_color(patch):
    if patch.size == 0:
        return "?"
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    if mask.sum() < 5:
        return "?"
    b, g, r = patch[mask].mean(axis=0)
    if r > g + 15 and r > b:
        return "RED"
    return "BLACK"


def match_rank(patch, template_set):
    best_name, best_score, _ = match_rank_detailed(patch, template_set)
    return best_name, best_score


def preprocess_rank_variants(patch, size=None):
    if patch.size == 0:
        return {}
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    variants = {}
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants["otsu_inv"] = otsu
    variants["adaptive_inv"] = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    variants["contrast_otsu_inv"] = cv2.threshold(
        cv2.equalizeHist(gray), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    if size is not None:
        variants = {
            name: cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
            for name, image in variants.items()
        }
    return variants


def _template_score(candidate, template):
    scores = []
    if template.shape == candidate.shape:
        result = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
        scores.append(float(result.max()))
    padded = cv2.copyMakeBorder(candidate, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    if template.shape[0] <= padded.shape[0] and template.shape[1] <= padded.shape[1]:
        result = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
        scores.append(float(result.max()))
    else:
        resized = cv2.resize(candidate, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        scores.append(float(result.max()))
    return max(scores) if scores else -1.0


def match_rank_detailed(patch, template_set):
    if patch.size == 0:
        return "?", 0.0, []
    best_name, best_score, best_variant = "?", -1.0, None
    candidates = {}
    for name, tmpl in template_set.items():
        scores = []
        for variant_name, variant in preprocess_rank_variants(patch).items():
            scores.append((_template_score(variant, tmpl), variant_name))
            normalized = cv2.resize(variant, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_NEAREST)
            scores.append((_template_score(normalized, tmpl), f"{variant_name}_normalized"))
        score, variant_name = max(scores, key=lambda item: item[0])
        candidates[name] = {"score": score, "variant": variant_name}
        if score > best_score:
            best_score = score
            best_name = name
            best_variant = variant_name
    ranked = [
        {"rank": name, "score": round(float(data["score"]), 4), "variant": data["variant"]}
        for name, data in sorted(candidates.items(), key=lambda item: item[1]["score"], reverse=True)
    ]
    return best_name, best_score, ranked


def _candidate_margin(candidates: list[dict[str, Any]]) -> float:
    if len(candidates) < 2:
        return 1.0
    return float(candidates[0]["score"]) - float(candidates[1]["score"])


def detect_ace_from_rank_corner(patch):
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = (gray < 180).astype("uint8") * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    candidates = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 250 or not (18 <= w <= 30 and 24 <= h <= 34 and 5 <= x <= 16 and 5 <= y <= 12):
            continue
        child_indices = [child_idx for child_idx, item in enumerate(hierarchy) if item[3] == idx]
        if len(child_indices) != 1:
            continue
        child = contours[child_indices[0]]
        cx, cy, cw, ch = cv2.boundingRect(child)
        child_area = cv2.contourArea(child)
        if not (5 <= cw <= 10 and 9 <= ch <= 15 and 25 <= child_area <= 70):
            continue
        if cy - y > 8:
            continue
        score = 0.96
        candidates.append({
            "rank": "A",
            "score": score,
            "variant": "corner_a_glyph_hole",
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "hole_bbox": {"x": cx, "y": cy, "w": cw, "h": ch},
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {
        "rank": "A",
        "score": candidates[0]["score"],
        "candidates": candidates,
    }


def assess_rank_attempts(attempts):
    usable = []
    exact_threshold = 0.96
    min_signal_score = 0.55
    min_margin = 0.08
    for attempt in attempts:
        source = attempt["source"]
        score = float(attempt["score"])
        margin = _candidate_margin(attempt.get("candidates", []))
        enriched = {**attempt, "margin": margin}
        if source == "ace_corner_detector":
            usable.append(enriched)
        elif source in ("full_last_card", "rank_corner"):
            if score >= exact_threshold:
                usable.append(enriched)
            elif score >= min_signal_score and margin >= min_margin:
                usable.append(enriched)
        elif source == "live_shape_heuristic" and score >= 0.88:
            usable.append(enriched)

    by_rank: dict[str, list[dict[str, Any]]] = {}
    for attempt in usable:
        by_rank.setdefault(attempt["rank"], []).append(attempt)

    exact = [
        attempt for attempt in usable
        if attempt["source"] in ("full_last_card", "rank_corner") and attempt["score"] >= exact_threshold
    ]
    if exact:
        exact.sort(key=lambda item: item["score"], reverse=True)
        best = exact[0]
        return best["rank"], best["score"], "template_confirmed", usable

    ace = by_rank.get("A", [])
    competing_rank_signal = any(
        attempt["rank"] != "A"
        and attempt["source"] in ("full_last_card", "rank_corner")
        and attempt["score"] >= min_signal_score
        for attempt in usable
    )
    if (
        ace
        and not competing_rank_signal
        and any(item["source"] == "ace_corner_detector" and item["score"] >= 0.95 for item in ace)
    ):
        best = max(ace, key=lambda item: item["score"])
        return "A", best["score"], "corner_glyph_confirmed", usable

    corroborated = [
        (rank, rank_attempts)
        for rank, rank_attempts in by_rank.items()
        if len({item["source"] for item in rank_attempts}) >= 2
    ]
    if corroborated:
        corroborated.sort(key=lambda item: max(attempt["score"] for attempt in item[1]), reverse=True)
        rank, rank_attempts = corroborated[0]
        if len(corroborated) > 1:
            return rank, max(item["score"] for item in rank_attempts), "conflicting_recognizers", usable
        best_score = max(item["score"] for item in rank_attempts)
        return rank, best_score, "template_confirmed", usable

    if len(by_rank) > 1:
        ranked = sorted(by_rank.items(), key=lambda item: max(attempt["score"] for attempt in item[1]), reverse=True)
        rank, rank_attempts = ranked[0]
        return rank, max(item["score"] for item in rank_attempts), "conflicting_recognizers", usable

    if usable:
        best = max(usable, key=lambda item: item["score"])
        provenance = "shape_heuristic_only" if best["source"] == "live_shape_heuristic" else "unresolved"
        return best["rank"], best["score"], provenance, usable

    if attempts:
        best = max(attempts, key=lambda item: item["score"])
        return best["rank"], best["score"], "unresolved", usable

    return "?", 0.0, "unresolved", usable


def recognize_card_rank(img, x, y, is_last, layout: BoardLayout):
    attempts = []
    if is_last:
        full_patch = _safe_crop(img, x, y, layout.last_card_crop_width, layout.last_card_crop_height)
        name, score, ranked = match_rank_detailed(full_patch, TEMPLATES_LAST)
        attempts.append({"source": "full_last_card", "rank": name, "score": score, "candidates": ranked})
        corner_patch = _safe_crop(img, x, y, layout.rank_width, layout.rank_height)
        ace = detect_ace_from_rank_corner(corner_patch)
        if ace is not None:
            attempts.append({
                "source": "ace_corner_detector",
                "rank": ace["rank"],
                "score": ace["score"],
                "candidates": ace["candidates"],
            })
        name, score, ranked = match_rank_detailed(corner_patch, TEMPLATES)
        attempts.append({"source": "rank_corner", "rank": name, "score": score, "candidates": ranked})
        heuristic = live_shape_rank_heuristic(full_patch)
        if heuristic is not None:
            attempts.append({
                "source": "live_shape_heuristic",
                "rank": heuristic["rank"],
                "score": heuristic["score"],
                "candidates": heuristic["candidates"],
            })
    else:
        corner_patch = _safe_crop(img, x, y, layout.rank_width, layout.rank_height)
        ace = detect_ace_from_rank_corner(corner_patch)
        if ace is not None:
            attempts.append({
                "source": "ace_corner_detector",
                "rank": ace["rank"],
                "score": ace["score"],
                "candidates": ace["candidates"],
            })
        name, score, ranked = match_rank_detailed(corner_patch, TEMPLATES)
        attempts.append({"source": "rank_corner", "rank": name, "score": score, "candidates": ranked})
    rank, score, provenance, usable = assess_rank_attempts(attempts)
    selected = max(
        [item for item in attempts if item["rank"] == rank] or attempts,
        key=lambda item: item["score"],
    )
    return rank, score, {
        "selected_source": selected["source"],
        "rank_provenance": provenance,
        "corroborating_sources": [item["source"] for item in usable if item["rank"] == rank],
        "attempts": [
            {
                "source": item["source"],
                "rank": item["rank"],
                "score": round(float(item["score"]), 4),
                "margin": round(float(_candidate_margin(item.get("candidates", []))), 4),
                "candidates": item["candidates"][:10],
            }
            for item in attempts
        ],
    }


def _rank_ink_mask(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 100, 110]))
    return cv2.morphologyEx(cv2.bitwise_or(cv2.bitwise_or(red1, red2), dark), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def live_shape_rank_heuristic(patch):
    if patch.size == 0:
        return None
    mask = _rank_ink_mask(patch)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 40:
            boxes.append({"x": x, "y": y, "w": w, "h": h, "area": float(area)})
    if not boxes:
        return None
    large = [box for box in boxes if box["y"] >= 35 and box["h"] >= 35]
    top = [box for box in boxes if box["y"] <= 8 and box["h"] >= 20]
    candidates = []

    widest = max(large, key=lambda box: box["w"], default=None)
    if widest and widest["w"] >= 55 and widest["h"] >= 40:
        score = min(0.92, 0.62 + (widest["w"] - 55) / 40 + widest["area"] / 10000)
        candidates.append({"rank": "Q", "score": round(float(score), 4), "variant": "wide_loop_shape"})

    lower_digits = sorted(
        [box for box in large if box["w"] >= 20 and box["area"] > 250],
        key=lambda box: box["x"],
    )
    top_wide = any(box["w"] >= 28 and box["h"] >= 24 for box in top)
    if len(lower_digits) >= 2 and top_wide:
        left, right = lower_digits[0], lower_digits[1]
        if left["x"] < right["x"] and left["w"] < right["w"] * 0.85:
            score = min(0.9, 0.64 + (right["area"] + left["area"]) / 12000)
            candidates.append({"rank": "10", "score": round(float(score), 4), "variant": "two_digit_shape"})

    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {
        "rank": candidates[0]["rank"],
        "score": candidates[0]["score"],
        "candidates": candidates,
    }


def classify_suit_color_detailed(patch):
    if patch.size == 0:
        return "?", [{"color": "RED", "score": 0.0}, {"color": "BLACK", "score": 0.0}]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    if mask.sum() < 5:
        return "?", [{"color": "RED", "score": 0.0}, {"color": "BLACK", "score": 0.0}]
    b, g, r = patch[mask].mean(axis=0)
    red_score = max(0.0, float((r - max(g, b)) / 255.0))
    dark_score = max(0.0, float((200.0 - gray[mask].mean()) / 200.0))
    black_score = dark_score * (1.0 - min(1.0, red_score * 3.0))
    candidates = [
        {"color": "RED", "score": round(red_score, 4), "mean_rgb": [round(float(r), 2), round(float(g), 2), round(float(b), 2)]},
        {"color": "BLACK", "score": round(black_score, 4), "mean_rgb": [round(float(r), 2), round(float(g), 2), round(float(b), 2)]},
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]["color"] if candidates[0]["score"] > 0 else "?", candidates


def default_transform_for_image(img: np.ndarray, layout: BoardLayout = REFERENCE_LAYOUT) -> BoardTransform:
    h, w = img.shape[:2]
    return BoardTransform(
        original_width=w,
        original_height=h,
        content_x=0,
        content_y=0,
        content_width=w,
        content_height=h,
        reference_width=layout.reference_width,
        reference_height=layout.reference_height,
    )


def transform_from_content_rect(
    image_width: int,
    image_height: int,
    content_rect: tuple[float, float, float, float],
    layout: BoardLayout = REFERENCE_LAYOUT,
) -> BoardTransform:
    x, y, w, h = content_rect
    return BoardTransform(image_width, image_height, x, y, w, h, layout.reference_width, layout.reference_height)


def _find_stack_boxes(img: np.ndarray, layout: BoardLayout) -> list[dict[str, Any] | None]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 75, 255]))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes_by_col: list[list[tuple[int, int, int, int]]] = [[] for _ in layout.tableau_x]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < layout.column_width * 0.35 or h < 25:
            continue
        if y < layout.slot_y + layout.slot_height + 80:
            continue
        if y > int(layout.reference_height * 0.88):
            continue
        cx = x + w / 2
        for idx, col_x in enumerate(layout.tableau_x):
            if col_x - 18 <= cx <= col_x + layout.column_width + 18:
                boxes_by_col[idx].append((x, y, w, h))
                break

    stack_boxes: list[dict[str, Any] | None] = []
    for boxes in boxes_by_col:
        if not boxes:
            stack_boxes.append(None)
            continue
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        stack_boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "bottom": y2})
    return stack_boxes


def _has_card_back_near_top(img: np.ndarray, box: dict[str, Any], layout: BoardLayout) -> bool:
    x1 = max(0, int(box["x"]))
    x2 = min(img.shape[1], int(box["x"] + box["w"]))
    y1 = max(0, int(box["y"]))
    y2 = min(img.shape[0], int(box["y"] + min(42, box["h"])))
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    saturated = cv2.inRange(hsv, np.array([95, 45, 45]), np.array([155, 255, 255]))
    return saturated.mean() > 12


def _estimate_hidden_step(stack_boxes: list[dict[str, Any] | None], img: np.ndarray, layout: BoardLayout) -> int:
    boxes = [b for b in stack_boxes if b]
    if not boxes:
        return layout.hidden_card_step
    top = min(b["y"] for b in boxes)
    offsets = sorted(
        int(round(b["y"] - top))
        for b in boxes
        if 12 <= b["y"] - top <= 160 and _has_card_back_near_top(img, b, layout)
    )
    diffs = [b - a for a, b in zip(offsets, offsets[1:]) if 12 <= b - a <= 45]
    if diffs:
        return int(round(float(np.median(diffs))))
    back_heights = sorted(
        int(round(b["h"]))
        for b in boxes
        if _has_card_back_near_top(img, b, layout) and b["h"] >= layout.full_card_height
    )
    height_diffs = [b - a for a, b in zip(back_heights, back_heights[1:]) if 12 <= b - a <= 45]
    if height_diffs:
        return int(round(float(np.median(height_diffs))))
    return layout.hidden_card_step


def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return img[y:y + h, x:x + w]


def _scan_rows_for_column(
    img: np.ndarray,
    col_idx: int,
    box: dict[str, Any] | None,
    board_bottom: int,
    layout: BoardLayout,
    hidden_step: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = {"column": col_idx, "box": box, "rows": [], "rejections": []}
    if not box or box["h"] < 80:
        return [], report

    global_top_candidates = [b["y"] for b in _find_stack_boxes(img, layout) if b]
    global_top = min(global_top_candidates) if global_top_candidates else layout.tableau_y_top
    x = int(round(layout.tableau_x[col_idx]))
    height = int(round(box["h"]))
    y_box = int(round(box["y"]))
    has_back = _has_card_back_near_top(img, box, layout)
    top_offset = max(0, int(round(y_box - global_top)))

    if top_offset > hidden_step / 2:
        hidden_count = int(round(top_offset / hidden_step))
        revealed_count = max(1, int(round((height - layout.full_card_height) / layout.revealed_card_step)) + 1)
        first_revealed_y = y_box
    elif has_back:
        hidden_count = max(1, int(round(max(0, height - layout.full_card_height) / max(1, hidden_step))))
        revealed_count = 1
        first_revealed_y = int(round(box["bottom"] - layout.full_card_height))
    else:
        hidden_count = 0
        revealed_count = max(1, int(round((height - layout.full_card_height) / layout.revealed_card_step)) + 1)
        first_revealed_y = y_box

    total = hidden_count + revealed_count
    if total > layout.max_tableau_cards:
        report["rejections"].append({"reason": "max_card_count", "count": total})
        return [], report

    cards: list[dict[str, Any]] = []
    row_positions: list[int] = []
    for idx in range(hidden_count):
        y = int(round(y_box + idx * hidden_step)) if has_back and top_offset <= hidden_step / 2 else int(round(global_top + idx * hidden_step))
        row_positions.append(y)
        cards.append({
            "rank": "?",
            "color": "?",
            "score": 0.0,
            "face_down": True,
            "provenance": "hidden_face_down",
            "bbox": {
                "x": x,
                "y": y,
                "w": layout.column_width,
                "h": hidden_step,
            },
        })
        report["rows"].append({"kind": "hidden", "y": y})

    for row in range(revealed_count):
        y = int(round(first_revealed_y + row * layout.revealed_card_step))
        if y + layout.full_card_height > board_bottom:
            report["rejections"].append({"reason": "row_leaves_board", "y": y})
            break
        if row_positions and y <= row_positions[-1]:
            report["rejections"].append({"reason": "non_monotonic_row", "y": y})
            break
        if row_positions and abs(y - row_positions[-1]) < 10:
            report["rejections"].append({"reason": "repeated_card_position", "y": y})
            break
        is_last = row == revealed_count - 1
        if is_last:
            name, score, rank_details = recognize_card_rank(img, x, y, is_last, layout)
            color_patch = _safe_crop(img, x + 55, y + 20, 35, 20)
            color, color_candidates = classify_suit_color_detailed(color_patch)
        else:
            name, score, rank_details = recognize_card_rank(img, x, y, is_last, layout)
            suit_patch = _safe_crop(img, x + layout.suit_x_offset, y + layout.suit_y_offset, layout.suit_width, layout.suit_height)
            color, color_candidates = classify_suit_color_detailed(suit_patch)
        if score < layout.min_rank_score:
            report["rejections"].append({"reason": "low_confidence_rank", "y": y, "rank": name, "score": float(score)})
            name, color, score = "?", "?", 0.0
        row_positions.append(y)
        card = {
            "rank": name,
            "color": color,
            "score": round(float(score), 2),
            "face_down": False,
            "provenance": "ocr_visible",
            "recognition": rank_details,
            "color_candidates": color_candidates,
            "bbox": {
                "x": x,
                "y": y,
                "w": layout.column_width,
                "h": layout.full_card_height if is_last else layout.revealed_card_step,
            },
        }
        cards.append(card)
        report["rows"].append({
            "kind": "revealed",
            "y": y,
            "rank": name,
            "color": color,
            "score": round(float(score), 3),
            "recognition": rank_details,
            "color_candidates": color_candidates,
        })

    if any(b <= a for a, b in zip(row_positions, row_positions[1:])):
        report["rejections"].append({"reason": "non_monotonic_rows", "rows": row_positions})
        return [], report
    return cards, report


def _detect_board_bottom(img: np.ndarray, layout: BoardLayout) -> int:
    # Stop before bottom controls/footers when they are visible. This is a guard
    # for scanning, not a coordinate transform.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 35, 20]), np.array([95, 255, 230]))
    row_fraction = green.mean(axis=1) / 255.0
    for y in range(img.shape[0] - 1, layout.tableau_y_top + 300, -1):
        if row_fraction[y] > 0.45:
            return min(img.shape[0], y + 1)
    return img.shape[0]


def _reject_impossible_multiplicities(board: dict[str, Any], report: dict[str, Any]) -> None:
    counts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for area in [f"col{i}" for i in range(7)] + ["free_cells", "foundation"]:
        cards = board.get(area, [])
        for idx, card in enumerate(cards):
            if not card or card.get("face_down") or card.get("rank") == "?" or card.get("color") == "?":
                continue
            key = (card["rank"], card["color"])
            counts.setdefault(key, []).append({"area": area, "index": idx})
    for (rank, color), positions in counts.items():
        if len(positions) > 2:
            keep = sorted(
                positions,
                key=lambda pos: board[pos["area"]][pos["index"]].get("score", 0.0),
                reverse=True,
            )[:2]
            keep_ids = {(pos["area"], pos["index"]) for pos in keep}
            report.setdefault("multiplicity_rejections", []).append({
                "rank": rank,
                "color": color,
                "positions": positions,
                "kept": keep,
            })
            for pos in positions:
                if (pos["area"], pos["index"]) in keep_ids:
                    continue
                card = board[pos["area"]][pos["index"]]
                card["rank"] = "?"
                card["color"] = "?"
                card["score"] = 0.0
                card["provenance"] = "rejected_impossible_multiplicity"


def build_observation(board: dict[str, Any]) -> BoardObservation:
    columns = []
    for idx in range(7):
        cards = board.get(f"col{idx}", [])
        hidden_count = sum(1 for card in cards if card and card.get("face_down"))
        visible_cards = tuple(card for card in cards if card and not card.get("face_down"))
        columns.append(ObservedColumn(hidden_count=hidden_count, visible_cards=visible_cards))
    return BoardObservation(
        columns=tuple(columns),
        free_cells=tuple(board.get("free_cells", [])),
        foundation=tuple(board.get("foundation", [])),
    )


def _detect_slot_y(img: np.ndarray, layout: BoardLayout, tableau_top: int) -> int:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 75, 255]))
    back = cv2.inRange(hsv, np.array([95, 45, 45]), np.array([155, 255, 255]))
    mask = cv2.bitwise_or(white, back)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 180 <= y < tableau_top - 55 and w >= 45 and h >= 45:
            candidates.append(y)
    return min(candidates) if candidates else layout.slot_y


def detect_column_height(img, x):
    layout = REFERENCE_LAYOUT
    boxes = _find_stack_boxes(img, layout)
    for idx, col_x in enumerate(layout.tableau_x):
        if abs(col_x - x) < 4 and boxes[idx]:
            box = boxes[idx]
            hidden_step = _estimate_hidden_step(boxes, img, layout)
            hidden_count = 0
            if _has_card_back_near_top(img, box, layout):
                hidden_count = max(0, int(round(max(0, box["h"] - layout.full_card_height) / max(1, hidden_step))))
            return int(box["h"]), hidden_count, True
    return 0, 0, True


def _load_image(image_or_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    img = cv2.imread(str(image_or_path))
    if img is None:
        raise FileNotFoundError(image_or_path)
    return img


def read_board(
    image_or_path,
    layout: BoardLayout | None = None,
    *,
    transform: BoardTransform | None = None,
    include_metadata: bool = False,
):
    layout = layout or REFERENCE_LAYOUT
    original = _load_image(image_or_path)
    transform = transform or default_transform_for_image(original, layout)
    img = transform.normalize_image(original)
    stack_boxes = _find_stack_boxes(img, layout)
    hidden_step = _estimate_hidden_step(stack_boxes, img, layout)
    tableau_tops = [b["y"] for b in stack_boxes if b]
    tableau_top = int(round(min(tableau_tops))) if tableau_tops else layout.tableau_y_top
    slot_y = _detect_slot_y(img, layout, tableau_top)
    calibrated_layout = replace(layout, hidden_card_step=hidden_step, tableau_y_top=tableau_top, slot_y=slot_y)
    board_bottom = _detect_board_bottom(img, calibrated_layout)

    board: dict[str, Any] = {}
    detection_report: dict[str, Any] = {
        "board_bottom": board_bottom,
        "hidden_card_step": hidden_step,
        "stack_boxes": stack_boxes,
        "columns": [],
    }
    for col_idx, box in enumerate(stack_boxes):
        cards, report = _scan_rows_for_column(img, col_idx, box, board_bottom, calibrated_layout, hidden_step)
        board[f"col{col_idx}"] = cards
        detection_report["columns"].append(report)

    def read_slot(x):
        patch = _safe_crop(img, x, calibrated_layout.slot_y, calibrated_layout.slot_width, calibrated_layout.slot_height)
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if gray.std() < 15:
            return None
        name, score = match_rank(patch, TEMPLATES_LAST)
        if score < calibrated_layout.min_rank_score:
            return None
        color = classify_suit_color(_safe_crop(img, x + 55, calibrated_layout.slot_y + 20, 35, 20))
        return {"rank": name, "color": color, "score": round(float(score), 2)}

    board["free_cells"] = [read_slot(x) for x in calibrated_layout.free_cell_x]
    board["foundation"] = [read_slot(x) for x in calibrated_layout.foundation_x]
    _reject_impossible_multiplicities(board, detection_report)

    if include_metadata:
        observation = build_observation(board)
        return {
            "board": board,
            "observation": observation,
            "layout": calibrated_layout,
            "transform": transform,
            "normalized_image": img,
            "detection_report": detection_report,
        }
    return board


def save_calibration_artifacts(image_or_path, calibration_dir, layout: BoardLayout | None = None) -> dict[str, Any]:
    layout = layout or REFERENCE_LAYOUT
    original = _load_image(image_or_path)
    result = read_board(original, layout=layout, include_metadata=True)
    out = Path(calibration_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "original.png"), original)
    normalized = result["normalized_image"]
    cv2.imwrite(str(out / "normalized.png"), normalized)

    overlay = normalized.copy()
    transform: BoardTransform = result["transform"]
    detection = result["detection_report"]
    board_bottom = detection["board_bottom"]
    cv2.rectangle(overlay, (0, 0), (layout.reference_width - 1, board_bottom - 1), (0, 255, 255), 2)
    cv2.putText(overlay, "content rect original=(%.1f,%.1f %.1fx%.1f) normalized=(0,0 %dx%d)" % (
        transform.content_x, transform.content_y, transform.content_width, transform.content_height,
        layout.reference_width, layout.reference_height,
    ), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    for idx, x in enumerate(layout.tableau_x):
        cv2.rectangle(overlay, (x, layout.tableau_y_top), (x + layout.column_width, board_bottom), (255, 200, 0), 1)
        cv2.putText(overlay, f"col{idx}", (x + 4, layout.tableau_y_top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        col_dir = out / "tableau_crops"
        col_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(col_dir / f"col{idx}.png"), normalized[layout.tableau_y_top:board_bottom, x:x + layout.column_width])

    rank_dir = out / "rank_crops"
    rank_dir.mkdir(exist_ok=True)
    for col in detection["columns"]:
        box = col.get("box")
        if box:
            cv2.circle(overlay, (int(box["x"] + box["w"] / 2), int(box["y"])), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (int(box["x"] + box["w"] / 2), int(box["bottom"])), 5, (0, 0, 255), -1)
        for row_idx, row in enumerate(col["rows"]):
            y = int(row["y"])
            x = layout.tableau_x[col["column"]]
            cv2.line(overlay, (x, y), (x + layout.column_width, y), (0, 128, 255), 2)
            cv2.putText(overlay, f"c{col['column']} r{row_idx} {row['kind']} y={y}", (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1)
            cv2.imwrite(str(rank_dir / f"col{col['column']}_row{row_idx}_{row['kind']}.png"),
                        _safe_crop(normalized, x, y, layout.last_card_crop_width, layout.last_card_crop_height))

    for name, xs in (("free_cell", layout.free_cell_x), ("foundation", layout.foundation_x)):
        crop_dir = out / f"{name}_crops"
        crop_dir.mkdir(exist_ok=True)
        for idx, x in enumerate(xs):
            cv2.rectangle(overlay, (x, layout.slot_y), (x + layout.slot_width, layout.slot_y + layout.slot_height), (255, 0, 255), 2)
            cv2.putText(overlay, f"{name}{idx}", (x + 2, layout.slot_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.imwrite(str(crop_dir / f"{idx}.png"), _safe_crop(normalized, x, layout.slot_y, layout.slot_width, layout.slot_height))

    cv2.imwrite(str(out / "layout_overlay.png"), overlay)
    (out / "layout.json").write_text(json.dumps(result["layout"].to_json_data(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    detection_payload = {
        **result["detection_report"],
        "transform": result["transform"].to_json_data(),
        "board": result["board"],
        "observation": result["observation"].to_json_data(),
    }
    (out / "detection_report.json").write_text(json.dumps(detection_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return detection_payload


def _write_threshold_artifacts(patch: np.ndarray, out: Path) -> None:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out / "threshold_otsu_inv.png"), preprocess_rank_variants(patch)["otsu_inv"])
    cv2.imwrite(str(out / "threshold_adaptive_inv.png"), preprocess_rank_variants(patch)["adaptive_inv"])
    cv2.imwrite(str(out / "threshold_contrast_otsu_inv.png"), preprocess_rank_variants(patch)["contrast_otsu_inv"])
    cv2.imwrite(str(out / "edges.png"), cv2.Canny(gray, 80, 160))


def save_exposed_card_diagnostics(
    image_or_path,
    column: int,
    index: int,
    output_dir,
    layout: BoardLayout | None = None,
) -> dict[str, Any]:
    layout = layout or REFERENCE_LAYOUT
    original = _load_image(image_or_path)
    result = read_board(original, layout=layout, include_metadata=True)
    calibrated_layout: BoardLayout = result["layout"]
    transform: BoardTransform = result["transform"]
    normalized = result["normalized_image"]
    report = result["detection_report"]["columns"][column]
    rows = report["rows"]
    if index < 0 or index >= len(rows):
        raise IndexError(f"Column {column} has {len(rows)} detected rows, cannot debug row {index}")
    row = rows[index]
    if row["kind"] != "revealed":
        raise ValueError(f"Column {column} row {index} is {row['kind']}, not an exposed card")

    x = int(calibrated_layout.tableau_x[column])
    y = int(row["y"])
    full_rect = (x, y, calibrated_layout.last_card_crop_width, calibrated_layout.full_card_height)
    rank_rect = (x, y, calibrated_layout.rank_width, calibrated_layout.rank_height)
    suit_rect = (x + 55, y + 20, 35, 20)
    full_card = _safe_crop(normalized, *full_rect)
    rank_crop = _safe_crop(normalized, *rank_rect)
    suit_crop = _safe_crop(normalized, *suit_rect)

    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "full_original_card_crop.png"), full_card)
    cv2.imwrite(str(out / "rank_crop.png"), rank_crop)
    cv2.imwrite(str(out / "suit_color_crop.png"), suit_crop)
    cv2.imwrite(str(out / "enlarged_nearest.png"), cv2.resize(full_card, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(out / "enlarged_interpolated.png"), cv2.resize(full_card, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    _write_threshold_artifacts(rank_crop, out)

    rank, score, rank_details = recognize_card_rank(normalized, x, y, True, calibrated_layout)
    color, color_candidates = classify_suit_color_detailed(suit_crop)
    current_card = result["board"][f"col{column}"][index]
    rejection_reason = None
    for rejection in report.get("rejections", []):
        if rejection.get("y") == y:
            rejection_reason = rejection
            break

    def map_rect(rect):
        rx, ry, rw, rh = rect
        x1, y1 = transform.normalized_to_original(rx, ry)
        x2, y2 = transform.normalized_to_original(rx + rw, ry + rh)
        return {
            "normalized": {"x": rx, "y": ry, "w": rw, "h": rh},
            "original": {
                "x": round(x1, 3),
                "y": round(y1, 3),
                "w": round(x2 - x1, 3),
                "h": round(y2 - y1, 3),
            },
        }

    coordinates = {
        "column": column,
        "index": index,
        "full_card": map_rect(full_rect),
        "rank": map_rect(rank_rect),
        "suit_color": map_rect(suit_rect),
    }
    recognition = {
        "current_recognition_result": current_card,
        "independent_rank_result": {"rank": rank, "score": round(float(score), 4), **rank_details},
        "top_10_rank_template_matches": next(
            attempt["candidates"][:10]
            for attempt in rank_details["attempts"]
            if attempt["source"] == rank_details["selected_source"]
        ),
        "all_rank_attempts": rank_details["attempts"],
        "top_color_suit_candidates": color_candidates,
        "selected_color": color,
        "rejection_reason": rejection_reason,
        "template_set_selected": "rank_corner + full_last_card",
        "template_set_reason": (
            "Live exposed card art uses clean rank glyphs; full_last_card templates include old-game artwork, "
            "so rank_corner is evaluated without overwriting existing templates."
        ),
    }
    (out / "crop_coordinates.json").write_text(json.dumps(coordinates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "recognition_result.json").write_text(json.dumps(recognition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "diagnostics.json").write_text(json.dumps({
        "coordinates": coordinates,
        "recognition": recognition,
        "layout": calibrated_layout.to_json_data(),
        "transform": transform.to_json_data(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "coordinates": coordinates,
        "recognition": recognition,
    }
