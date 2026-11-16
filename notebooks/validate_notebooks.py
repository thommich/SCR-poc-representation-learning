"""Execute all notebooks in synthetic demo mode without changing their cells."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[1]
for name in ("00_extract_and_month_vectors.ipynb", "01_temporal_models.ipynb",
             "02_geometry_and_lgbm.ipynb"):
    path = root / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    print("Executed", name)
