# 3D Bin Packing

Cleaned repository for 3D container loading experiments using heuristic and MIP approaches.

## Repository Layout

- `src/`: Python scripts used for model/algorithm prototyping.
- `notebooks/heuristic/`: Heuristic-focused Jupyter notebooks.
- `notebooks/mip/`: MIP-focused Jupyter notebooks.
- `notebooks/experiments/`: Miscellaneous experiment notebooks.
- `assets/images/`: Visualization images.
- `data/outputs/`: Large text outputs and generated run logs.
- `docs/`: Project docs and static pages.
- `web/`: Streamlit web app for upload, packing, and 3D visualization.

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run scripts from `src/` or open notebooks from `notebooks/`.

## Web App

Run the app locally:

```powershell
streamlit run web/app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Open `https://share.streamlit.io` and create a new app.
3. Select your repo and branch.
4. Set **Main file path** to `web/app.py`.
5. Set the dependencies file to `web/requirements.txt`.
6. Deploy and, if needed, set app visibility to **Public** to share with anyone via link.

What the app does:

- accepts an uploaded Excel file (`.xlsx`)
- reads `items` and `bins` sheets
- computes a packing plan with a heuristic engine
- shows KPI tables plus a 3D interactive loading view
- supports downloading results as Excel

Input format:

- `items` sheet required columns: `item`, `length`, `width`, `height`
- optional item columns: `weight`, `quantity`, `allow_rotation`
- `bins` sheet required columns: `bin`, `length`, `width`, `height`
- optional bin columns: `max_weight`, `quantity`

You can download an input template directly from the web app.

## Notes

- `src/3d_loading.py` currently contains a hardcoded local Excel file path. Update it before running in a new environment.
- Large generated outputs are kept under `data/outputs/` and ignored by Git.
