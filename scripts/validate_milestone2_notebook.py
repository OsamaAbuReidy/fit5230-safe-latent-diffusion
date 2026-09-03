"""Execute Milestone 2 notebook code cells sequentially without Jupyter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "milestone2_show_of_force.ipynb"


def main() -> int:
    matplotlib.use("Agg")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_validation__"}
    previous = Path.cwd()
    os.chdir(ROOT)
    executed = 0
    try:
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
            except Exception as error:
                raise RuntimeError(f"Notebook code cell {index} failed") from error
            executed += 1
    finally:
        os.chdir(previous)
    print(f"Validated {executed} code cells from {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
