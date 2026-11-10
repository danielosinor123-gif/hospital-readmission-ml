"""Execute a notebook and report exactly which cell fails and why.

Usage:  python src/run_notebook.py [path/to/notebook.ipynb]
"""
import sys
import traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
      else PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb")

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}})
try:
    client.execute()
    print("OK - all cells executed")
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:200])
    # find the cell that errored
    for i, cell in enumerate(nb["cells"]):
        if cell.get("outputs"):
            for out in cell["outputs"]:
                if out.get("output_type") == "error":
                    print(f"\n=== cell {i} raised {out.get('ename')}: {out.get('evalue')} ===")
                    src = "".join(cell["source"])
                    print("--- source (first 40 lines) ---")
                    print("\n".join(src.splitlines()[:40]))
                    print("--- traceback (last 15 lines) ---")
                    tb = "\n".join(out.get("traceback", []))
                    print("\n".join(tb.splitlines()[-15:]))
                    raise SystemExit(1)
nbformat.write(nb, NB)
print("saved executed notebook")
