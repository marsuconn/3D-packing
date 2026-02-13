from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import io
from math import isfinite
from typing import Any

import pandas as pd

EPSILON = 1e-9


@dataclass
class PackingResult:
    placements: pd.DataFrame
    unpacked: pd.DataFrame
    bin_summary: pd.DataFrame
    metrics: dict[str, Any]


ITEM_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "item": ("item", "item_id", "sku", "id", "name"),
    "length": ("length", "l"),
    "width": ("width", "w"),
    "height": ("height", "h"),
    "weight": ("weight", "wt"),
    "quantity": ("quantity", "qty", "count"),
    "allow_rotation": ("allow_rotation", "rotation", "can_rotate"),
}

BIN_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "bin": ("bin", "bin_id", "container", "container_id", "truck", "name"),
    "length": ("length", "l"),
    "width": ("width", "w"),
    "height": ("height", "h"),
    "max_weight": ("max_weight", "weight_capacity", "capacity_weight"),
    "quantity": ("quantity", "qty", "count"),
}


def run_packing(items_df: pd.DataFrame, bins_df: pd.DataFrame) -> PackingResult:
    items_prepared = _prepare_items(items_df)
    bins_prepared = _prepare_bins(bins_df)

    expanded_items = _expand_items(items_prepared)
    expanded_bins = _expand_bins(bins_prepared)

    bin_states = [_init_bin_state(bin_row) for bin_row in expanded_bins.to_dict(orient="records")]

    placements: list[dict[str, Any]] = []
    unpacked: list[dict[str, Any]] = []

    for item in expanded_items.to_dict(orient="records"):
        was_placed = False
        for bin_state in bin_states:
            placement = _try_place_item(bin_state, item)
            if placement is not None:
                placements.append(placement)
                was_placed = True
                break
        if not was_placed:
            unpacked.append(
                {
                    "item_id": item["item_id"],
                    "source_item": item["source_item"],
                    "length": item["length"],
                    "width": item["width"],
                    "height": item["height"],
                    "weight": item["weight"],
                    "volume": item["volume"],
                    "reason": "No feasible space found in available bins.",
                }
            )

    placements_df = pd.DataFrame(
        placements,
        columns=[
            "item_id",
            "source_item",
            "bin_id",
            "bin_type",
            "x",
            "y",
            "z",
            "length",
            "width",
            "height",
            "weight",
            "volume",
            "orientation",
        ],
    )
    unpacked_df = pd.DataFrame(
        unpacked,
        columns=["item_id", "source_item", "length", "width", "height", "weight", "volume", "reason"],
    )

    bin_summary_df = _build_bin_summary(bin_states)
    metrics = _build_metrics(expanded_items, placements_df, unpacked_df, bin_summary_df)
    return PackingResult(
        placements=placements_df,
        unpacked=unpacked_df,
        bin_summary=bin_summary_df,
        metrics=metrics,
    )


def read_excel_input(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    required_sheets = {"items", "bins"}
    missing = required_sheets.difference(workbook.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required sheet(s): {missing_list}.")
    return workbook["items"], workbook["bins"]


def build_template_workbook() -> bytes:
    items_template = pd.DataFrame(
        [
            {"item": "A", "length": 4, "width": 3, "height": 2, "weight": 10, "quantity": 3, "allow_rotation": True},
            {"item": "B", "length": 6, "width": 2, "height": 2, "weight": 8, "quantity": 2, "allow_rotation": True},
            {"item": "C", "length": 3, "width": 3, "height": 3, "weight": 12, "quantity": 1, "allow_rotation": False},
        ]
    )
    bins_template = pd.DataFrame(
        [
            {"bin": "Truck-XL", "length": 12, "width": 8, "height": 8, "max_weight": 500, "quantity": 1},
            {"bin": "Truck-L", "length": 10, "width": 6, "height": 6, "max_weight": 350, "quantity": 1},
        ]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        items_template.to_excel(writer, index=False, sheet_name="items")
        bins_template.to_excel(writer, index=False, sheet_name="bins")
    return buffer.getvalue()


def result_to_workbook(result: PackingResult) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.placements.to_excel(writer, index=False, sheet_name="placements")
        result.unpacked.to_excel(writer, index=False, sheet_name="unpacked")
        result.bin_summary.to_excel(writer, index=False, sheet_name="bin_summary")
        pd.DataFrame([result.metrics]).to_excel(writer, index=False, sheet_name="metrics")
    return buffer.getvalue()


def _prepare_items(items_df: pd.DataFrame) -> pd.DataFrame:
    items = _rename_columns(items_df.copy(), ITEM_COLUMN_ALIASES)
    _require_columns(items, ("item", "length", "width", "height"), entity_name="items")

    items["item"] = items["item"].astype(str).str.strip()
    if (items["item"] == "").any():
        raise ValueError("`items` sheet has empty item identifiers.")

    for column in ("length", "width", "height"):
        items[column] = pd.to_numeric(items[column], errors="coerce")
    _ensure_positive(items, ("length", "width", "height"), entity_name="items")

    if "weight" not in items.columns:
        items["weight"] = 0.0
    items["weight"] = pd.to_numeric(items["weight"], errors="coerce").fillna(0.0)
    if (items["weight"] < 0).any():
        raise ValueError("`items.weight` must be non-negative.")

    if "quantity" in items.columns:
        items["quantity"] = _parse_quantity(items["quantity"], "items.quantity")
    else:
        items["quantity"] = 1

    if "allow_rotation" not in items.columns:
        items["allow_rotation"] = True
    items["allow_rotation"] = items["allow_rotation"].apply(_as_bool)

    return items[["item", "length", "width", "height", "weight", "quantity", "allow_rotation"]]


def _prepare_bins(bins_df: pd.DataFrame) -> pd.DataFrame:
    bins = _rename_columns(bins_df.copy(), BIN_COLUMN_ALIASES)
    _require_columns(bins, ("bin", "length", "width", "height"), entity_name="bins")

    bins["bin"] = bins["bin"].astype(str).str.strip()
    if (bins["bin"] == "").any():
        raise ValueError("`bins` sheet has empty bin identifiers.")

    for column in ("length", "width", "height"):
        bins[column] = pd.to_numeric(bins[column], errors="coerce")
    _ensure_positive(bins, ("length", "width", "height"), entity_name="bins")

    if "max_weight" not in bins.columns:
        bins["max_weight"] = float("inf")
    bins["max_weight"] = pd.to_numeric(bins["max_weight"], errors="coerce")
    bins.loc[bins["max_weight"].isna(), "max_weight"] = float("inf")
    finite_mask = bins["max_weight"].apply(isfinite)
    invalid_weight_capacity = bins.loc[finite_mask & (bins["max_weight"] <= 0)]
    if not invalid_weight_capacity.empty:
        raise ValueError("`bins.max_weight` must be positive for finite values.")

    if "quantity" in bins.columns:
        bins["quantity"] = _parse_quantity(bins["quantity"], "bins.quantity")
    else:
        bins["quantity"] = 1
    return bins[["bin", "length", "width", "height", "max_weight", "quantity"]]


def _expand_items(items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in items.to_dict(orient="records"):
        quantity = int(row["quantity"])
        for index in range(1, quantity + 1):
            item_id = row["item"] if quantity == 1 else f"{row['item']}#{index}"
            length = float(row["length"])
            width = float(row["width"])
            height = float(row["height"])
            rows.append(
                {
                    "item_id": item_id,
                    "source_item": row["item"],
                    "length": length,
                    "width": width,
                    "height": height,
                    "weight": float(row["weight"]),
                    "allow_rotation": bool(row["allow_rotation"]),
                    "volume": length * width * height,
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("`items` sheet has no usable rows.")
    return expanded.sort_values(by=["volume", "weight"], ascending=[False, False], ignore_index=True)


def _expand_bins(bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in bins.to_dict(orient="records"):
        quantity = int(row["quantity"])
        for index in range(1, quantity + 1):
            bin_id = row["bin"] if quantity == 1 else f"{row['bin']}#{index}"
            length = float(row["length"])
            width = float(row["width"])
            height = float(row["height"])
            rows.append(
                {
                    "bin_id": bin_id,
                    "bin_type": row["bin"],
                    "length": length,
                    "width": width,
                    "height": height,
                    "max_weight": float(row["max_weight"]),
                    "capacity_volume": length * width * height,
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("`bins` sheet has no usable rows.")
    return expanded.reset_index(drop=True)


def _init_bin_state(bin_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bin_id": bin_row["bin_id"],
        "bin_type": bin_row["bin_type"],
        "length": bin_row["length"],
        "width": bin_row["width"],
        "height": bin_row["height"],
        "max_weight": bin_row["max_weight"],
        "used_weight": 0.0,
        "placements": [],
        "free_spaces": [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "length": bin_row["length"],
                "width": bin_row["width"],
                "height": bin_row["height"],
            }
        ],
    }


def _try_place_item(bin_state: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    projected_weight = bin_state["used_weight"] + item["weight"]
    if isfinite(bin_state["max_weight"]) and projected_weight > (bin_state["max_weight"] + EPSILON):
        return None

    candidates: list[tuple[tuple[float, float, float, float], int, tuple[float, float, float, str]]] = []
    for space_index, space in enumerate(bin_state["free_spaces"]):
        for orientation in _orientations(item):
            item_length, item_width, item_height, orientation_name = orientation
            if _fits(space, item_length, item_width, item_height):
                leftover_volume = _space_volume(space) - item["volume"]
                score = (leftover_volume, space["z"], space["y"], space["x"])
                candidates.append((score, space_index, (item_length, item_width, item_height, orientation_name)))

    if not candidates:
        return None

    _, chosen_index, chosen_orientation = min(candidates, key=lambda candidate: candidate[0])
    item_length, item_width, item_height, orientation_name = chosen_orientation

    chosen_space = bin_state["free_spaces"].pop(chosen_index)
    placement = {
        "item_id": item["item_id"],
        "source_item": item["source_item"],
        "bin_id": bin_state["bin_id"],
        "bin_type": bin_state["bin_type"],
        "x": chosen_space["x"],
        "y": chosen_space["y"],
        "z": chosen_space["z"],
        "length": item_length,
        "width": item_width,
        "height": item_height,
        "weight": item["weight"],
        "volume": item["volume"],
        "orientation": orientation_name,
    }
    bin_state["used_weight"] = projected_weight
    bin_state["placements"].append(placement)

    new_spaces = _split_space(chosen_space, item_length, item_width, item_height)
    bin_state["free_spaces"].extend(new_spaces)
    bin_state["free_spaces"] = _prune_free_spaces(bin_state["free_spaces"])
    return placement


def _orientations(item: dict[str, Any]) -> list[tuple[float, float, float, str]]:
    axes = (("L", item["length"]), ("W", item["width"]), ("H", item["height"]))
    if not item["allow_rotation"]:
        return [(float(item["length"]), float(item["width"]), float(item["height"]), "LWH")]

    variants: list[tuple[float, float, float, str]] = []
    seen: set[tuple[float, float, float]] = set()
    for perm in permutations(axes, 3):
        dimensions = (float(perm[0][1]), float(perm[1][1]), float(perm[2][1]))
        if dimensions in seen:
            continue
        seen.add(dimensions)
        orientation_name = "".join(axis[0] for axis in perm)
        variants.append((dimensions[0], dimensions[1], dimensions[2], orientation_name))
    return variants


def _fits(space: dict[str, float], item_length: float, item_width: float, item_height: float) -> bool:
    return (
        item_length <= space["length"] + EPSILON
        and item_width <= space["width"] + EPSILON
        and item_height <= space["height"] + EPSILON
    )


def _split_space(
    space: dict[str, float],
    item_length: float,
    item_width: float,
    item_height: float,
) -> list[dict[str, float]]:
    split_spaces = [
        {
            "x": space["x"] + item_length,
            "y": space["y"],
            "z": space["z"],
            "length": space["length"] - item_length,
            "width": space["width"],
            "height": space["height"],
        },
        {
            "x": space["x"],
            "y": space["y"] + item_width,
            "z": space["z"],
            "length": item_length,
            "width": space["width"] - item_width,
            "height": space["height"],
        },
        {
            "x": space["x"],
            "y": space["y"],
            "z": space["z"] + item_height,
            "length": item_length,
            "width": item_width,
            "height": space["height"] - item_height,
        },
    ]
    return [
        candidate
        for candidate in split_spaces
        if candidate["length"] > EPSILON and candidate["width"] > EPSILON and candidate["height"] > EPSILON
    ]


def _prune_free_spaces(spaces: list[dict[str, float]]) -> list[dict[str, float]]:
    cleaned = [
        candidate
        for candidate in spaces
        if candidate["length"] > EPSILON and candidate["width"] > EPSILON and candidate["height"] > EPSILON
    ]

    pruned: list[dict[str, float]] = []
    for index, candidate in enumerate(cleaned):
        is_contained = False
        for other_index, other in enumerate(cleaned):
            if index == other_index:
                continue
            if _is_contained(candidate, other):
                is_contained = True
                break
        if not is_contained:
            pruned.append(candidate)

    return sorted(pruned, key=lambda space: (space["z"], space["y"], space["x"], _space_volume(space)))


def _is_contained(inner: dict[str, float], outer: dict[str, float]) -> bool:
    return (
        inner["x"] >= outer["x"] - EPSILON
        and inner["y"] >= outer["y"] - EPSILON
        and inner["z"] >= outer["z"] - EPSILON
        and inner["x"] + inner["length"] <= outer["x"] + outer["length"] + EPSILON
        and inner["y"] + inner["width"] <= outer["y"] + outer["width"] + EPSILON
        and inner["z"] + inner["height"] <= outer["z"] + outer["height"] + EPSILON
    )


def _space_volume(space: dict[str, float]) -> float:
    return float(space["length"] * space["width"] * space["height"])


def _build_bin_summary(bin_states: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in bin_states:
        capacity_volume = state["length"] * state["width"] * state["height"]
        loaded_volume = sum(placement["volume"] for placement in state["placements"])
        packed_items = len(state["placements"])
        max_weight = state["max_weight"]
        finite_weight = isfinite(max_weight)
        rows.append(
            {
                "bin_id": state["bin_id"],
                "bin_type": state["bin_type"],
                "length": state["length"],
                "width": state["width"],
                "height": state["height"],
                "capacity_volume": capacity_volume,
                "max_weight": max_weight,
                "packed_items": packed_items,
                "loaded_volume": loaded_volume,
                "used_weight": state["used_weight"],
                "volume_utilization": (loaded_volume / capacity_volume) if capacity_volume > 0 else 0.0,
                "weight_utilization": (state["used_weight"] / max_weight) if finite_weight and max_weight > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def _build_metrics(
    expanded_items: pd.DataFrame,
    placements: pd.DataFrame,
    unpacked: pd.DataFrame,
    bin_summary: pd.DataFrame,
) -> dict[str, Any]:
    total_items = int(len(expanded_items))
    packed_items = int(len(placements))
    unpacked_items = int(len(unpacked))

    total_item_volume = float(expanded_items["volume"].sum()) if "volume" in expanded_items.columns else 0.0
    packed_volume = float(placements["volume"].sum()) if not placements.empty else 0.0
    total_bin_volume = float(bin_summary["capacity_volume"].sum()) if not bin_summary.empty else 0.0

    bins_used = int((bin_summary["packed_items"] > 0).sum()) if "packed_items" in bin_summary.columns else 0
    used_bin_capacity = (
        float(bin_summary.loc[bin_summary["packed_items"] > 0, "capacity_volume"].sum())
        if not bin_summary.empty
        else 0.0
    )

    return {
        "total_items": total_items,
        "packed_items": packed_items,
        "unpacked_items": unpacked_items,
        "packing_rate": (packed_items / total_items) if total_items else 0.0,
        "total_item_volume": total_item_volume,
        "packed_volume": packed_volume,
        "global_volume_utilization": (packed_volume / total_bin_volume) if total_bin_volume > 0 else 0.0,
        "used_bin_volume_utilization": (packed_volume / used_bin_capacity) if used_bin_capacity > 0 else 0.0,
        "total_bins": int(len(bin_summary)),
        "bins_used": bins_used,
    }


def _rename_columns(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    normalized_map = {_normalize_column_name(column): column for column in df.columns}
    rename_map: dict[str, str] = {}
    for canonical_name, options in aliases.items():
        for alias in options:
            normalized_alias = _normalize_column_name(alias)
            if normalized_alias in normalized_map:
                rename_map[normalized_map[normalized_alias]] = canonical_name
                break
    return df.rename(columns=rename_map)


def _normalize_column_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], entity_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"`{entity_name}` is missing required column(s): {missing_names}.")


def _ensure_positive(df: pd.DataFrame, columns: tuple[str, ...], entity_name: str) -> None:
    if df[list(columns)].isna().any().any():
        raise ValueError(f"`{entity_name}` contains non-numeric values in: {', '.join(columns)}.")
    invalid_rows = df[(df[list(columns)] <= 0).any(axis=1)]
    if not invalid_rows.empty:
        raise ValueError(f"`{entity_name}` has non-positive values in dimensions: {', '.join(columns)}.")


def _parse_quantity(source: pd.Series, label: str) -> pd.Series:
    raw = pd.to_numeric(source, errors="coerce")
    raw = raw.fillna(1)
    invalid = (raw < 1) | ((raw % 1) != 0)
    if invalid.any():
        raise ValueError(f"`{label}` must be a positive integer.")
    return raw.astype(int)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"0", "false", "f", "no", "n"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    return True
