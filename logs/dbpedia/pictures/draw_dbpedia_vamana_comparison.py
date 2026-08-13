#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/kai3/coco/hnsw_rabitq")
OUT = ROOT / "logs/dbpedia/pictures"

METHODS = [
    {
        "key": "raw_hnsw",
        "name": "Raw HNSW",
        "path": Path("/home/kai3/coco/hnsw/build/hnsw_dbpedia_ef_400_M_32_omp64.log"),
        "color": "#4C78A8",
        "marker": "o",
    },
    {
        "key": "hnsw_rabitq_d",
        "name": "HNSW-RaBitQ D",
        "path": ROOT / "logs/dbpedia/dbpedia_D_ef_400_M_32.log",
        "color": "#F58518",
        "marker": "s",
    },
    {
        "key": "vamana_refine2",
        "name": "Vamana ExRaBitQ4 refine2",
        "path": ROOT / "logs/dbpedia/vamana/dbpedia_vamana_Lbuild_400_M_32_refine_2.log",
        "color": "#54A24B",
        "marker": "^",
    },
]


def grab_float(pattern, text, default=None):
    match = re.search(pattern, text)
    if not match:
        return default
    return float(match.group(1))


def parse_kv_tail(line):
    values = {}
    for key, value in re.findall(r"([A-Za-z0-9_@]+)=([^\s]+)", line):
        cleaned = value.strip().rstrip(",)")
        try:
            values[key] = float(cleaned)
        except ValueError:
            values[key] = cleaned
    return values


def parse_log(method):
    text = method["path"].read_text(encoding="utf-8", errors="replace")
    graph_s = grab_float(r"Graph construction time:\s*([0-9.eE+-]+)\s*seconds", text)
    build_s = grab_float(r"Build time:\s*([0-9.eE+-]+)\s*seconds", text)
    if graph_s is None:
        graph_s = grab_float(r"build_stage=vamana_graph_build\s+seconds=([0-9.eE+-]+)", text)
    if build_s is None:
        build_s = grab_float(r"build_time_seconds=([0-9.eE+-]+)", text)
    storage_mb = grab_float(r"Index storage size:\s*([0-9.eE+-]+)\s*MB", text)

    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("L_search="):
            l_match = re.search(r"L_search=([0-9]+)", line)
            r_match = re.search(r"recall@10=([0-9.eE+-]+)", line)
            us_match = re.search(r"total_us_per_query_us=([0-9.eE+-]+)", line)
            qps_match = re.search(r"\bQPS=([0-9.eE+-]+)", line)
            if not (l_match and r_match and us_match):
                continue
            search_width = int(l_match.group(1))
            recall = float(r_match.group(1))
            latency_us = float(us_match.group(1))
            qps = float(qps_match.group(1)) if qps_match else 1e6 / latency_us
            kv = parse_kv_tail(line)
            points.append({
                "method": method["key"],
                "method_name": method["name"],
                "search_width": search_width,
                "recall": recall,
                "latency_us": latency_us,
                "qps": qps,
                "visited_nodes": kv.get("avg_visited_nodes", ""),
                "distance_computations": kv.get("avg_distance_computations", ""),
                "paper_prune_ratio": kv.get("paper_prune_ratio", ""),
                "paper_saved_ratio": kv.get("paper_saved_ratio", ""),
            })
            continue
        if re.match(r"^[0-9]+\s+[0-9.]+\s+[0-9.eE+-]+\s+us\b", line):
            parts = line.split()
            search_width = int(parts[0])
            recall = float(parts[1])
            latency_us = float(parts[2])
            kv = parse_kv_tail(line)
            qps = kv.get("qps", 1e6 / latency_us)
            points.append({
                "method": method["key"],
                "method_name": method["name"],
                "search_width": search_width,
                "recall": recall,
                "latency_us": latency_us,
                "qps": qps,
                "visited_nodes": kv.get("visited_nodes", ""),
                "distance_computations": kv.get("distance_computations", ""),
                "paper_prune_ratio": kv.get("paper_prune_ratio", ""),
                "paper_saved_ratio": kv.get("paper_saved_ratio", ""),
            })

    return {
        "method": method,
        "graph_s": graph_s,
        "build_s": build_s,
        "storage_mb": storage_mb,
        "points": sorted(points, key=lambda p: p["search_width"]),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_at_target(points, target):
    for point in sorted(points, key=lambda p: p["search_width"]):
        if point["recall"] >= target:
            return point
    return max(points, key=lambda p: p["recall"]) if points else None


def plot_build_storage(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metrics = [
        ("graph_s", "Graph construction time", "seconds"),
        ("build_s", "Total build time", "seconds"),
        ("storage_mb", "Index storage", "MB"),
    ]
    names = [r["method"]["name"] for r in results]
    colors = [r["method"]["color"] for r in results]
    for ax, (key, title, ylabel) in zip(axes, metrics):
        values = [r[key] for r in results]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("DBpedia build and storage comparison")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"build_storage_bars.{ext}", dpi=180)
    plt.close(fig)


def plot_recall_qps(results):
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for r in results:
        pts = [p for p in r["points"] if p["recall"] >= 0.90]
        ax.plot(
            [p["recall"] for p in pts],
            [p["qps"] for p in pts],
            marker=r["method"]["marker"],
            color=r["method"]["color"],
            linewidth=1.8,
            markersize=4,
            label=r["method"]["name"],
        )
    ax.axvline(0.995, color="#999999", linestyle="--", linewidth=1, label="recall@10=0.995")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS")
    ax.set_xlim(0.90, 1.000)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("DBpedia Recall@10 vs QPS")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"recall_vs_qps.{ext}", dpi=180)
    plt.close(fig)


def plot_effort_curves(results):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    panels = [
        ("recall", "Recall@10"),
        ("qps", "QPS"),
        ("latency_us", "Latency (us/query)"),
    ]
    for ax, (key, ylabel) in zip(axes, panels):
        for r in results:
            pts = r["points"]
            ax.plot(
                [p["search_width"] for p in pts],
                [p[key] for p in pts],
                marker=r["method"]["marker"],
                color=r["method"]["color"],
                linewidth=1.8,
                markersize=4,
                label=r["method"]["name"],
            )
        if key == "recall":
            ax.axhline(0.995, color="#999999", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Search width (efSearch or L_search)")
    axes[0].legend(loc="lower right")
    fig.suptitle("DBpedia search-width curves")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ef_curves.{ext}", dpi=180)
    plt.close(fig)


def plot_paper_prune(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, constrained_layout=True)
    for ax, key, title in [
        (axes[0], "paper_prune_ratio", "paper_prune_ratio"),
        (axes[1], "paper_saved_ratio", "paper_saved_ratio"),
    ]:
        for r in results:
            pts = [
                p for p in r["points"]
                if isinstance(p.get(key), float) and p["search_width"] >= 10
            ]
            if not pts:
                continue
            ax.plot(
                [p["search_width"] for p in pts],
                [p[key] for p in pts],
                marker=r["method"]["marker"],
                color=r["method"]["color"],
                linewidth=1.8,
                markersize=4,
                label=r["method"]["name"],
            )
        ax.set_title(title)
        ax.set_xlabel("Search width (efSearch or L_search)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ratio")
    axes[1].legend()
    fig.suptitle("Paper-prune effectiveness")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"paper_prune_curves.{ext}", dpi=180)
    plt.close(fig)


def write_report(results):
    target = 0.995
    rows = []
    for r in results:
        best = max(r["points"], key=lambda p: p["recall"])
        target_point = best_at_target(r["points"], target)
        rows.append((r, best, target_point))

    lines = [
        "# DBpedia Vamana / HNSW comparison",
        "",
        "Sources:",
        "",
    ]
    for r in results:
        lines.append(f"- `{r['method']['path']}`")
    lines.extend([
        "",
        "## Summary",
        "",
        "| Method | Graph build s | Total build s | Storage MB | Best recall | Best width | Best QPS | First width >= 0.995 | QPS there | Latency us there |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r, best, target_point in rows:
        first_width = target_point["search_width"] if target_point else ""
        first_qps = f"{target_point['qps']:.3f}" if target_point else ""
        first_lat = f"{target_point['latency_us']:.3f}" if target_point else ""
        lines.append(
            f"| {r['method']['name']} | {r['graph_s']:.3f} | {r['build_s']:.3f} | "
            f"{r['storage_mb']:.2f} | {best['recall']:.5f} | {best['search_width']} | "
            f"{best['qps']:.3f} | {first_width} | {first_qps} | {first_lat} |"
        )

    raw = next(r for r in results if r["method"]["key"] == "raw_hnsw")
    d = next(r for r in results if r["method"]["key"] == "hnsw_rabitq_d")
    v = next(r for r in results if r["method"]["key"] == "vamana_refine2")
    lines.extend([
        "",
        "## Key ratios",
        "",
        f"- HNSW-RaBitQ D storage / raw HNSW storage: `{d['storage_mb'] / raw['storage_mb']:.3f}x`.",
        f"- Vamana storage / raw HNSW storage: `{v['storage_mb'] / raw['storage_mb']:.3f}x`.",
        f"- Vamana total build / HNSW-RaBitQ D total build: `{v['build_s'] / d['build_s']:.3f}x`.",
        "",
        "## Generated files",
        "",
        "- `summary.csv`",
        "- `eval_points.csv`",
        "- `common_search_width_comparison.csv`",
        "- `build_storage_bars.png/pdf`",
        "- `recall_vs_qps.png/pdf`",
        "- `ef_curves.png/pdf`",
        "- `paper_prune_curves.png/pdf`",
        "",
    ])
    (OUT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [parse_log(m) for m in METHODS]
    for r in results:
        if not r["points"]:
            raise RuntimeError(f"no eval points parsed from {r['method']['path']}")
        for key in ("graph_s", "build_s", "storage_mb"):
            if r[key] is None or math.isnan(r[key]):
                raise RuntimeError(f"missing {key} in {r['method']['path']}")

    summary_rows = []
    eval_rows = []
    for r in results:
        best = max(r["points"], key=lambda p: p["recall"])
        at_995 = best_at_target(r["points"], 0.995)
        summary_rows.append({
            "method": r["method"]["key"],
            "method_name": r["method"]["name"],
            "graph_build_seconds": r["graph_s"],
            "total_build_seconds": r["build_s"],
            "storage_mb": r["storage_mb"],
            "best_recall": best["recall"],
            "best_search_width": best["search_width"],
            "best_qps": best["qps"],
            "first_search_width_recall_ge_0995": at_995["search_width"] if at_995 else "",
            "qps_at_first_recall_ge_0995": at_995["qps"] if at_995 else "",
            "latency_us_at_first_recall_ge_0995": at_995["latency_us"] if at_995 else "",
        })
        eval_rows.extend(r["points"])

    fields = [
        "method", "method_name", "graph_build_seconds", "total_build_seconds",
        "storage_mb", "best_recall", "best_search_width", "best_qps",
        "first_search_width_recall_ge_0995", "qps_at_first_recall_ge_0995",
        "latency_us_at_first_recall_ge_0995",
    ]
    write_csv(OUT / "summary.csv", summary_rows, fields)
    write_csv(
        OUT / "eval_points.csv",
        eval_rows,
        [
            "method", "method_name", "search_width", "recall", "latency_us",
            "qps", "visited_nodes", "distance_computations",
            "paper_prune_ratio", "paper_saved_ratio",
        ],
    )

    by_method_width = {
        r["method"]["key"]: {p["search_width"]: p for p in r["points"]}
        for r in results
    }
    common_widths = sorted(set.intersection(*[
        set(items.keys()) for items in by_method_width.values()
    ]))
    common_rows = []
    for width in common_widths:
        row = {"search_width": width}
        for r in results:
            p = by_method_width[r["method"]["key"]][width]
            row[f"{r['method']['key']}_recall"] = p["recall"]
            row[f"{r['method']['key']}_qps"] = p["qps"]
            row[f"{r['method']['key']}_latency_us"] = p["latency_us"]
        common_rows.append(row)
    common_fields = ["search_width"]
    for r in results:
        common_fields.extend([
            f"{r['method']['key']}_recall",
            f"{r['method']['key']}_qps",
            f"{r['method']['key']}_latency_us",
        ])
    write_csv(OUT / "common_search_width_comparison.csv", common_rows, common_fields)

    plot_build_storage(results)
    plot_recall_qps(results)
    plot_effort_curves(results)
    plot_paper_prune(results)
    write_report(results)


if __name__ == "__main__":
    main()
