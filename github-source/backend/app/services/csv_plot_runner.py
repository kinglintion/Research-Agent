"""Child process for validated plotting snippets. No application credentials are passed in."""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(sys.argv[1])
    with (root / "input.csv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        rows = list(reader)
    safe = {name: getattr(__builtins__, name) for name in
            ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max",
             "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip")}
    namespace = {"__builtins__": safe, "rows": rows, "columns": columns, "plt": plt, "np": np, "math": math}
    exec(compile((root / "plot_code.py").read_text(encoding="utf-8"), "plot_code.py", "exec"), namespace)
    fig = namespace.get("fig") or plt.gcf()
    fig.savefig(root / "figure.png", format="png", dpi=160, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
