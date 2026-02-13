from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

# Ensure sibling package imports (e.g. `src.*`) work when Streamlit runs from `web/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.packing_engine import PackingResult, build_template_workbook, result_to_workbook, run_packing


st.set_page_config(page_title="3D Bin Packing", layout="wide")


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)


@st.cache_data(show_spinner=False)
def template_bytes() -> bytes:
    return build_template_workbook()


def build_box_mesh(
    x: float,
    y: float,
    z: float,
    length: float,
    width: float,
    height: float,
    color: str,
    hover_label: str,
) -> go.Mesh3d:
    vx = [x, x + length, x + length, x, x, x + length, x + length, x]
    vy = [y, y, y + width, y + width, y, y, y + width, y + width]
    vz = [z, z, z, z, z + height, z + height, z + height, z + height]

    return go.Mesh3d(
        x=vx,
        y=vy,
        z=vz,
        i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 0, 1, 2],
        j=[1, 2, 5, 6, 1, 2, 3, 0, 4, 3, 5, 6],
        k=[2, 3, 6, 7, 5, 6, 7, 4, 5, 7, 6, 7],
        color=color,
        opacity=0.7,
        name=hover_label,
        hovertemplate=f"{hover_label}<extra></extra>",
        showscale=False,
        showlegend=False,
    )


def build_bin_wireframe(length: float, width: float, height: float) -> go.Scatter3d:
    corners = [
        (0, 0, 0),
        (length, 0, 0),
        (length, width, 0),
        (0, width, 0),
        (0, 0, height),
        (length, 0, height),
        (length, width, height),
        (0, width, height),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in edges:
        x_values.extend([corners[start][0], corners[end][0], None])
        y_values.extend([corners[start][1], corners[end][1], None])
        z_values.extend([corners[start][2], corners[end][2], None])

    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line={"color": "#111827", "width": 5},
        name="Bin",
        hoverinfo="skip",
        showlegend=False,
    )


def build_bin_figure(selected_bin: str, result: PackingResult) -> go.Figure:
    bin_row = result.bin_summary.loc[result.bin_summary["bin_id"] == selected_bin].iloc[0]
    placements = result.placements.loc[result.placements["bin_id"] == selected_bin]

    fig = go.Figure()
    fig.add_trace(build_bin_wireframe(float(bin_row["length"]), float(bin_row["width"]), float(bin_row["height"])))

    unique_source_items = placements["source_item"].drop_duplicates().tolist()
    palette = qualitative.Bold + qualitative.Safe + qualitative.Vivid + qualitative.Dark24
    color_map = {item: palette[index % len(palette)] for index, item in enumerate(unique_source_items)}

    for placement in placements.to_dict(orient="records"):
        hover_label = (
            f"Item: {placement['item_id']}<br>"
            f"Bin: {placement['bin_id']}<br>"
            f"Pos: ({placement['x']:.2f}, {placement['y']:.2f}, {placement['z']:.2f})<br>"
            f"Dims: ({placement['length']:.2f}, {placement['width']:.2f}, {placement['height']:.2f})"
        )
        fig.add_trace(
            build_box_mesh(
                x=float(placement["x"]),
                y=float(placement["y"]),
                z=float(placement["z"]),
                length=float(placement["length"]),
                width=float(placement["width"]),
                height=float(placement["height"]),
                color=color_map[placement["source_item"]],
                hover_label=hover_label,
            )
        )

    max_dimension = max(float(bin_row["length"]), float(bin_row["width"]), float(bin_row["height"]))
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis_title": "Length (X)",
            "yaxis_title": "Width (Y)",
            "zaxis_title": "Height (Z)",
            "xaxis": {"range": [0, float(bin_row["length"])]},
            "yaxis": {"range": [0, float(bin_row["width"])]},
            "zaxis": {"range": [0, float(bin_row["height"])]},
            "aspectmode": "manual",
            "aspectratio": {
                "x": float(bin_row["length"]) / max_dimension,
                "y": float(bin_row["width"]) / max_dimension,
                "z": float(bin_row["height"]) / max_dimension,
            },
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
        },
    )
    return fig


def render_result(result: PackingResult, runtime_seconds: float) -> None:
    st.subheader("Results")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Total Items", int(result.metrics["total_items"]))
    metric_cols[1].metric("Packed", int(result.metrics["packed_items"]))
    metric_cols[2].metric("Unpacked", int(result.metrics["unpacked_items"]))
    metric_cols[3].metric("Packing Rate", f"{result.metrics['packing_rate'] * 100:.1f}%")
    metric_cols[4].metric("Bins Used", f"{int(result.metrics['bins_used'])}/{int(result.metrics['total_bins'])}")
    metric_cols[5].metric("Runtime (s)", f"{runtime_seconds:.2f}")

    st.markdown("**Bin Summary**")
    st.dataframe(result.bin_summary, use_container_width=True)

    if not result.placements.empty:
        bins_with_items = result.bin_summary.loc[result.bin_summary["packed_items"] > 0, "bin_id"].tolist()
        default_bin_options = bins_with_items if bins_with_items else result.bin_summary["bin_id"].tolist()
        selected_bin = st.selectbox("Select a bin for 3D visualization", default_bin_options, index=0)
        st.plotly_chart(build_bin_figure(selected_bin, result), use_container_width=True)
        st.markdown("**Placements**")
        st.dataframe(result.placements, use_container_width=True)
    else:
        st.warning("No items could be packed with the provided bins.")

    if not result.unpacked.empty:
        st.markdown("**Unpacked Items**")
        st.dataframe(result.unpacked, use_container_width=True)

    st.download_button(
        label="Download Results (Excel)",
        data=result_to_workbook(result),
        file_name="packing_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.title("3D Bin Packing Web App")
    st.write(
        "Upload an `.xlsx` file with two sheets: `items` and `bins`. "
        "The app will compute a packing plan and render a 3D loading view."
    )

    st.download_button(
        label="Download Input Template",
        data=template_bytes(),
        file_name="packing_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is None:
        st.info("Upload a file to start.")
        return

    file_bytes = uploaded_file.getvalue()
    workbook = load_workbook(file_bytes)
    sheet_lookup = {sheet_name.strip().lower(): sheet_name for sheet_name in workbook.keys()}
    missing_sheets = [name for name in ("items", "bins") if name not in sheet_lookup]
    if missing_sheets:
        st.error(f"Missing required sheet(s): {', '.join(missing_sheets)}.")
        return

    items_sheet_name = sheet_lookup["items"]
    bins_sheet_name = sheet_lookup["bins"]
    items_df = workbook[items_sheet_name]
    bins_df = workbook[bins_sheet_name]

    preview_columns = st.columns(2)
    with preview_columns[0]:
        st.markdown("**Items Preview**")
        st.dataframe(items_df.head(15), use_container_width=True)
    with preview_columns[1]:
        st.markdown("**Bins Preview**")
        st.dataframe(bins_df.head(15), use_container_width=True)

    if st.button("Run Packing", type="primary"):
        start_time = time.perf_counter()
        try:
            result = run_packing(items_df, bins_df)
        except ValueError as error:
            st.error(str(error))
            return
        runtime_seconds = time.perf_counter() - start_time
        st.session_state["packing_result"] = result
        st.session_state["runtime_seconds"] = runtime_seconds

    cached_result = st.session_state.get("packing_result")
    if cached_result is not None:
        render_result(cached_result, float(st.session_state.get("runtime_seconds", 0.0)))


if __name__ == "__main__":
    main()
