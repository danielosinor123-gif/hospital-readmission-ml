"""Verify the executed notebook: cell counts, figures, and sample outputs."""
import nbformat
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "01_data_understanding.ipynb"
nb = nbformat.read(NB, as_version=4)

code = [c for c in nb["cells"] if c["cell_type"] == "code"]
md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
imgs = sum(1 for c in code for o in c.get("outputs", [])
           if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]

print(f"markdown cells: {len(md)} | code cells: {len(code)} | figures: {imgs} | errors: {len(errs)}")

# Show a one-line sample of printed output from a few key cells
shown = 0
for i, c in enumerate(code):
    texts = ["".join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream"]
    if texts:
        first = texts[0].splitlines()[0] if texts[0].splitlines() else ""
        print(f"  cell {i:2d}: {first[:95]}")
        shown += 1
    if shown >= 12:
        break
