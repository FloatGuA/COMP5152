"""
Dataset filter & classifier for COMP5152 project.
Reads first 1/5 of CSVs from etfs/ and stocks/, classifies into A/B/C/D grades,
copies files into output/{grade}/{category}/ with _SPARSE suffix if density < 1.0.

Grade thresholds (BOTH conditions must hold):
  A: year_span >= 10  AND rows >= 2500
  B: year_span >= 5   AND rows >= 1250
  C: year_span >= 3   AND rows >= 750
  D: everything else

Density = rows / (year_span * 252). If < 1.0, filename gets _SPARSE suffix.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("C:/Coding/COMP5152/archive")
OUTPUT_DIR = Path("C:/Coding/COMP5152/output")
CATEGORIES = ["etfs", "stocks"]

GRADE_THRESHOLDS = [
    ("A", 10, 2500),
    ("B", 5,  1250),
    ("C", 3,  750),
]

TRADING_DAYS_PER_YEAR = 252


def get_grade(year_span: float, rows: int) -> str:
    for grade, min_years, min_rows in GRADE_THRESHOLDS:
        if year_span >= min_years and rows >= min_rows:
            return grade
    return "D"


def analyze_csv(filepath: Path) -> dict | None:
    """Read only first and last line to extract date range and row count."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # lines[0] is header
    data_lines = [l for l in lines[1:] if l.strip()]
    if len(data_lines) < 2:
        return None

    try:
        first_date = datetime.strptime(data_lines[0].split(",")[0], "%Y-%m-%d")
        last_date  = datetime.strptime(data_lines[-1].split(",")[0], "%Y-%m-%d")
    except ValueError:
        return None

    rows = len(data_lines)
    year_span = (last_date - first_date).days / 365.25
    density = rows / (year_span * TRADING_DAYS_PER_YEAR) if year_span > 0 else 0

    return {
        "rows": rows,
        "year_span": round(year_span, 2),
        "density": round(density, 3),
        "first_date": first_date.strftime("%Y-%m-%d"),
        "last_date": last_date.strftime("%Y-%m-%d"),
    }


def build_output_name(stem: str, density: float) -> str:
    suffix = "_SPARSE" if density < 1.0 else ""
    return f"{stem}{suffix}.csv"


def main():
    stats = {"A": 0, "B": 0, "C": 0, "D": 0, "skipped": 0, "sparse": 0}

    for category in CATEGORIES:
        src_dir = BASE_DIR / category
        all_files = sorted(src_dir.glob("*.csv"))
        subset = all_files[: max(1, len(all_files) // 5)]
        print(f"\n[{category}] total={len(all_files)}, processing first 1/5 = {len(subset)}")

        for csv_path in subset:
            info = analyze_csv(csv_path)
            if info is None:
                stats["skipped"] += 1
                print(f"  SKIP  {csv_path.name} (parse error or too few rows)")
                continue

            grade = get_grade(info["year_span"], info["rows"])
            stats[grade] += 1
            if info["density"] < 1.0:
                stats["sparse"] += 1

            dest_dir = OUTPUT_DIR / grade / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            out_name = build_output_name(csv_path.stem, info["density"])
            dest_path = dest_dir / out_name
            shutil.copy2(csv_path, dest_path)

            print(
                f"  [{grade}] {out_name:<30} "
                f"rows={info['rows']:>5}  "
                f"years={info['year_span']:>5}  "
                f"density={info['density']:.3f}  "
                f"{info['first_date']} → {info['last_date']}"
            )

    print("\n=== Summary ===")
    for grade in ["A", "B", "C", "D"]:
        print(f"  Grade {grade}: {stats[grade]}")
    print(f"  Sparse (density<1): {stats['sparse']}")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
