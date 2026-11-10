"""
Download the "Diabetes 130-US hospitals for years 1999-2008" dataset
from the UCI Machine Learning Repository.

WHY A SCRIPT AND NOT A MANUAL DOWNLOAD?
- Reproducibility: anyone (including your reviewer) can re-create `data/raw/`
  by running this one file.
- Documentation: the exact source URL and date are recorded.

Dataset reference:
Strack B., DeShazo J., Gennings C., et al. (2014).
"Impact of HbA1c Measurement on Hospital Readmission Rates:
 Analysis of 70,000 Clinical Database Patient Records."
BioMed Research International, vol. 2014, Article ID 781670.
https://doi.org/10.1155/2014/781670
"""

import zipfile
from pathlib import Path
from urllib.request import urlopen

# Paths are relative to THIS file so the script works from any working directory.
# __file__ = .../hospital_readmission_ml/src/download_data.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Official UCI static download link for dataset id 296.
URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
ZIP_PATH = DATA_RAW / "diabetes_130_us_hospitals_1999_2008.zip"

def download(url: str, dest: Path) -> None:
    """Download `url` to `dest`, skipping if the file already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded ({dest.stat().st_size:,} bytes)")
        return
    print(f"[download] {url} ...")
    with urlopen(url, timeout=120) as response, open(dest, "wb") as f:
        f.write(response.read())
    print(f"[done] saved -> {dest} ({dest.stat().st_size:,} bytes)")

def unzip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract a zip file and return the list of extracted file paths."""
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
    with zipfile.ZipFile(zip_path) as zf:
        extracted = zf.namelist()
        zf.extractall(out_dir)
    print("[extract]", extracted)
    return [out_dir / name for name in extracted]

def main() -> None:
    download(URL, ZIP_PATH)
    files = unzip(ZIP_PATH, DATA_RAW)
    print("\nFiles in data/raw/:")
    for f in DATA_RAW.rglob("*"):
        if f.is_file():
            print(f"  {f.relative_to(PROJECT_ROOT)}  ({f.stat().st_size:,} bytes)")

if __name__ == "__main__":          # only runs when executed directly, not on import
    main()
