#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    numeric = {
        "recall", "qps", "visited_nodes", "distance_computations", "ef",
        "centroid_count", "active_centroids", "cycles_per_distance"
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] not in ("", "unavailable"):
                row[key] = float(row[key])
    return rows


def label(row):
    return f"K={int(row['centroid_count'])} {row['layout']} {row['centroid_mode']}"


def plot_group(rows, x, y, output):
    groups = {}
    for row in rows:
        if isinstance(row.get(x), float) and isinstance(row.get(y), float):
            groups.setdefault(label(row), []).append(row)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in sorted(groups.items()):
        values.sort(key=lambda item: item[x])
        ax.plot([item[x] for item in values], [item[y] for item in values], marker="o", label=name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/multimeans_turbo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv)
    for x, y in [
        ("recall", "qps"),
        ("recall", "visited_nodes"),
        ("recall", "distance_computations"),
        ("ef", "recall"),
        ("centroid_count", "active_centroids"),
        ("centroid_count", "cycles_per_distance"),
    ]:
        plot_group(rows, x, y, args.output_dir / f"{x}_vs_{y}.png")


if __name__ == "__main__":
    main()
