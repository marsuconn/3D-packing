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

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run scripts from `src/` or open notebooks from `notebooks/`.

## Notes

- `src/3d_loading.py` currently contains a hardcoded local Excel file path. Update it before running in a new environment.
- Large generated outputs are kept under `data/outputs/` and ignored by Git.
