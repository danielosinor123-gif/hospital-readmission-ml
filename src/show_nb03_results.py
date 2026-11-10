"""Print the printed outputs of notebook 03 (no images) for review."""
import nbformat
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "03_tuning_threshold_and_explainability.ipynb"
nb = nbformat.read(NB, as_version=4)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    texts = ["".join(o.get("text", [])) for o in cell.get("outputs", [])
             if o.get("output_type") == "stream"]
    if texts:
        joined = "\n".join(texts).strip()
        if joined:
            print(f"===== cell {i} =====")
            print(joined[:1600])
            print()
