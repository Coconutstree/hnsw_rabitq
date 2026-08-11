#!/usr/bin/env python3

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


COLORS = {
    "hnsw": "#4C78A8",
    "rabitq_hnsw": "#F58518",
}
MARKERS = {
    "hnsw": "o",
    "rabitq_hnsw": "s",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)


EVAL_RE = re.compile(
    r"^\s*(?P<ef>\d+)\s+"
    r"(?P<recall>\d+(?:\.\d+)?)\s+"
    r"(?P<latency_us>\d+(?:\.\d+)?)\s+us\b"
)
GRAPH_TIME_RE = re.compile(r"^Graph construction time:\s*(?P<seconds>\d+(?:\.\d+)?)\s+seconds")
BUILD_TIME_RE = re.compile(r"^Build time:\s*(?P<seconds>\d+(?:\.\d+)?)\s+seconds")
INDEX_STORAGE_RE = re.compile(
    r"^Index storage size:\s*(?P<total_mb>\d+(?:\.\d+)?)\s+MB\s*"
    r"\(index=(?P<index_mb>\d+(?:\.\d+)?)\s+MB,\s*"
    r"auxiliary=(?P<aux_mb>\d+(?:\.\d+)?)\s+MB"
    r"(?:,\s*residual=(?P<residual_mb>\d+(?:\.\d+)?)\s+MB)?"
)
STORAGE_BREAKDOWN_RE = re.compile(
    r"\bstorage_breakdown\b.*?"
    r"index_file_MB=(?P<index_file_mb>\d+(?:\.\d+)?)\s+"
    r"auxiliary_MB=(?P<auxiliary_mb>\d+(?:\.\d+)?)\s+"
    r"raw_vector_inside_index_MB=(?P<raw_vector_inside_index_mb>\d+(?:\.\d+)?)\s+"
    r"graph_and_metadata_estimate_MB=(?P<graph_and_metadata_estimate_mb>\d+(?:\.\d+)?)\s+"
    r"extra_rerank_data_MB=(?P<extra_rerank_data_mb>\d+(?:\.\d+)?)\s+"
    r"effective_query_storage_MB=(?P<effective_query_storage_mb>\d+(?:\.\d+)?)"
)


def parse_key_values(line: str) -> dict[str, str]:
    values = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value.rstrip(",)")
    return values


def parse_log(path: Path, label: str) -> dict:
    dataset = {
        "label": label,
        "path": path,
        "graph_construction_seconds": None,
        "build_seconds": None,
        "storage_total_mb": None,
        "storage_index_mb": None,
        "storage_auxiliary_mb": 0.0,
        "storage_residual_mb": 0.0,
        "storage_breakdown": {},
        "points": [],
    }

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        match = GRAPH_TIME_RE.match(line)
        if match:
            dataset["graph_construction_seconds"] = float(match.group("seconds"))
            continue

        match = BUILD_TIME_RE.match(line)
        if match:
            dataset["build_seconds"] = float(match.group("seconds"))
            continue

        match = INDEX_STORAGE_RE.match(line)
        if match:
            dataset["storage_total_mb"] = float(match.group("total_mb"))
            dataset["storage_index_mb"] = float(match.group("index_mb"))
            dataset["storage_auxiliary_mb"] = float(match.group("aux_mb"))
            dataset["storage_residual_mb"] = float(match.group("residual_mb") or 0.0)
            continue

        match = STORAGE_BREAKDOWN_RE.search(line)
        if match:
            dataset["storage_breakdown"] = {
                key: float(value) for key, value in match.groupdict().items()
            }
            continue

        match = EVAL_RE.match(line)
        if match:
            key_values = parse_key_values(line)
            total_us = float(key_values.get("total_us_per_query", match.group("latency_us")))
            hnsw_us = float(key_values.get("hnsw_search_us_per_query", match.group("latency_us")))
            qps = float(key_values.get("qps", 1_000_000.0 / total_us))
            dataset["points"].append(
                {
                    "label": label,
                    "ef": int(match.group("ef")),
                    "recall": float(match.group("recall")),
                    "latency_us": total_us,
                    "hnsw_search_us": hnsw_us,
                    "qps": qps,
                    "p95_us": float(key_values["p95_us"]) if "p95_us" in key_values else "",
                    "rerank_us": (
                        float(key_values["residual_rerank_us_per_query"])
                        if "residual_rerank_us_per_query" in key_values
                        else ""
                    ),
                }
            )

    required = ["graph_construction_seconds", "build_seconds", "storage_total_mb"]
    missing = [key for key in required if dataset[key] is None]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    if not dataset["points"]:
        raise ValueError(f"{path} has no evaluation rows")

    dataset["points"].sort(key=lambda point: point["ef"])
    return dataset


def write_csvs(datasets: list[dict], output_dir: Path) -> None:
    summary_fields = [
        "label",
        "log_path",
        "graph_construction_seconds",
        "build_seconds",
        "storage_total_mb",
        "storage_index_mb",
        "storage_auxiliary_mb",
        "storage_residual_mb",
        "best_recall",
        "best_recall_ef",
        "best_recall_qps",
        "fastest_ef",
        "fastest_recall",
        "fastest_qps",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for dataset in datasets:
            best_recall = max(dataset["points"], key=lambda point: point["recall"])
            fastest = max(dataset["points"], key=lambda point: point["qps"])
            writer.writerow(
                {
                    "label": dataset["label"],
                    "log_path": dataset["path"],
                    "graph_construction_seconds": dataset["graph_construction_seconds"],
                    "build_seconds": dataset["build_seconds"],
                    "storage_total_mb": dataset["storage_total_mb"],
                    "storage_index_mb": dataset["storage_index_mb"],
                    "storage_auxiliary_mb": dataset["storage_auxiliary_mb"],
                    "storage_residual_mb": dataset["storage_residual_mb"],
                    "best_recall": best_recall["recall"],
                    "best_recall_ef": best_recall["ef"],
                    "best_recall_qps": best_recall["qps"],
                    "fastest_ef": fastest["ef"],
                    "fastest_recall": fastest["recall"],
                    "fastest_qps": fastest["qps"],
                }
            )

    point_fields = [
        "label",
        "ef",
        "recall",
        "latency_us",
        "qps",
        "hnsw_search_us",
        "rerank_us",
        "p95_us",
    ]
    with (output_dir / "eval_points.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=point_fields)
        writer.writeheader()
        for dataset in datasets:
            writer.writerows(dataset["points"])

    common_efs = sorted(set.intersection(*(set(point["ef"] for point in d["points"]) for d in datasets)))
    by_label = {
        dataset["label"]: {point["ef"]: point for point in dataset["points"]}
        for dataset in datasets
    }
    common_fields = ["ef"]
    for dataset in datasets:
        common_fields.extend(
            [
                f"{dataset['label']}_recall",
                f"{dataset['label']}_qps",
                f"{dataset['label']}_latency_us",
            ]
        )
    if len(datasets) == 2:
        common_fields.extend(["recall_delta_second_minus_first", "qps_ratio_second_over_first"])
    with (output_dir / "common_ef_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=common_fields)
        writer.writeheader()
        first_label = datasets[0]["label"]
        second_label = datasets[1]["label"] if len(datasets) > 1 else None
        for ef in common_efs:
            row = {"ef": ef}
            for dataset in datasets:
                point = by_label[dataset["label"]][ef]
                row[f"{dataset['label']}_recall"] = point["recall"]
                row[f"{dataset['label']}_qps"] = point["qps"]
                row[f"{dataset['label']}_latency_us"] = point["latency_us"]
            if second_label:
                row["recall_delta_second_minus_first"] = (
                    by_label[second_label][ef]["recall"] - by_label[first_label][ef]["recall"]
                )
                row["qps_ratio_second_over_first"] = (
                    by_label[second_label][ef]["qps"] / by_label[first_label][ef]["qps"]
                )
            writer.writerow(row)


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_summary_bars(datasets: list[dict], output_dir: Path) -> None:
    labels = [dataset["label"] for dataset in datasets]
    xs = range(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    metrics = [
        ("Graph construction", "Time (s)", [d["graph_construction_seconds"] for d in datasets]),
        ("End-to-end build", "Time (s)", [d["build_seconds"] for d in datasets]),
        ("Index storage", "Storage (GB)", [d["storage_total_mb"] / 1024.0 for d in datasets]),
    ]
    for ax, (title, ylabel, values) in zip(axes, metrics):
        bars = ax.bar(
            xs,
            values,
            width=0.58,
            color=[COLORS.get(label, "#666666") for label in labels],
            edgecolor="#222222",
            linewidth=0.7,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(xs), labels)
        ax.tick_params(axis="x", rotation=12)
        clean_axis(ax)
        annotate_bars(ax, bars)

        if len(values) == 2:
            if "storage" in title.lower():
                ratio_text = f"-{(1.0 - values[1] / values[0]) * 100.0:.1f}%"
            else:
                ratio_text = f"{values[1] / values[0]:.2f}x"
            ax.text(
                0.5,
                max(values) * 1.12,
                ratio_text,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_ylim(0, max(values) * 1.32)
    fig.tight_layout()
    save_figure(fig, output_dir, "build_storage_bars")


def plot_recall_qps(datasets: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for dataset in datasets:
        points = sorted(
            (point for point in dataset["points"] if point["recall"] >= 0.9),
            key=lambda point: point["recall"],
        )
        if not points:
            continue
        ax.plot(
            [point["recall"] for point in points],
            [point["qps"] for point in points],
            marker=MARKERS.get(dataset["label"], "o"),
            linewidth=2.1,
            markersize=4.8,
            color=COLORS.get(dataset["label"]),
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=dataset["label"],
        )
    ax.set_xlabel("Recall@1")
    ax.set_ylabel("QPS")
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    clean_axis(ax)
    ax.grid(True, axis="x", alpha=0.18, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, "recall_vs_qps")


def plot_ef_curves(datasets: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for dataset in datasets:
        points = dataset["points"]
        axes[0].plot(
            [point["ef"] for point in points],
            [point["recall"] for point in points],
            marker=MARKERS.get(dataset["label"], "o"),
            linewidth=2,
            markersize=4,
            color=COLORS.get(dataset["label"]),
            label=dataset["label"],
        )
        axes[1].plot(
            [point["ef"] for point in points if point["qps"] <= 3000.0],
            [point["qps"] for point in points if point["qps"] <= 3000.0],
            marker=MARKERS.get(dataset["label"], "o"),
            linewidth=2,
            markersize=4,
            color=COLORS.get(dataset["label"]),
            label=dataset["label"],
        )
    axes[0].set_xlabel("efSearch")
    axes[0].set_ylabel("Recall@1")
    axes[0].set_title("Recall vs efSearch")
    axes[1].set_xlabel("efSearch")
    axes[1].set_ylabel("QPS")
    axes[1].set_title("QPS vs efSearch")
    axes[1].set_ylim(0, 3000)
    for ax in axes:
        clean_axis(ax)
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.8)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "ef_curves")


def write_report(datasets: list[dict], output_dir: Path) -> None:
    lines = [
        "# DBpedia HNSW vs RaBitQ Case C 对比",
        "",
        "## 对比方式",
        "",
        "- 构图时间、总构建时间、有效查询存储占用都是单值指标，使用柱状图对比。",
        "- 查询性能以 Recall@1 vs QPS 为主图，因为它能在相近准确率下比较吞吐；图中只展示 recall >= 0.9 的点，横坐标固定为 0.9-1.0。",
        "- efSearch 曲线作为辅助诊断图，用来观察 search effort 增加时 recall 和 QPS 的变化；QPS 子图固定为 0-3000，超过该范围的点不画。",
        "- QPS 优先读取日志中的 `qps=`；原始 HNSW 日志没有该字段时，按 `1e6 / total_us_per_query` 换算。",
        "- 日志中没有运行时 RSS/峰值内存字段，因此这里的“内存占用”采用 `Index storage size` 表示有效查询存储占用。",
        "",
        "## 摘要",
        "",
        "| 方法 | 构图时间 s | 总构建 s | 存储占用 MB | 最高 recall | 最高 recall ef | 最高 recall QPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        best = max(dataset["points"], key=lambda point: point["recall"])
        lines.append(
            f"| {dataset['label']} | "
            f"{dataset['graph_construction_seconds']:.3f} | "
            f"{dataset['build_seconds']:.3f} | "
            f"{dataset['storage_total_mb']:.2f} | "
            f"{best['recall']:.5f} | "
            f"{best['ef']} | "
            f"{best['qps']:.2f} |"
        )

    if len(datasets) == 2:
        first, second = datasets
        graph_ratio = second["graph_construction_seconds"] / first["graph_construction_seconds"]
        build_ratio = second["build_seconds"] / first["build_seconds"]
        storage_ratio = second["storage_total_mb"] / first["storage_total_mb"]
        lines.extend(
            [
                "",
                "## 相对结果",
                "",
                f"- `{second['label']}` 的构图时间是 `{first['label']}` 的 `{graph_ratio:.3f}x`。",
                f"- `{second['label']}` 的总构建时间是 `{first['label']}` 的 `{build_ratio:.3f}x`。",
                f"- `{second['label']}` 的有效查询存储占用是 `{first['label']}` 的 `{storage_ratio:.3f}x`，"
                f"降低 `{(1.0 - storage_ratio) * 100.0:.1f}%`。",
                "",
                "## 生成文件",
                "",
                "- `summary.csv`: 每个方法一行的总体指标。",
                "- `eval_points.csv`: 解析出的全部 ef/recall/latency/QPS 明细。",
                "- `common_ef_comparison.csv`: 相同 ef 下的 recall 差值和 QPS 比值。",
                "- `build_storage_bars.png/pdf`: 构图时间、总构建时间、存储占用柱状图。",
                "- `recall_vs_qps.png/pdf`: 主要的准确率-吞吐对比图。",
                "- `ef_curves.png/pdf`: efSearch 辅助诊断曲线。",
            ]
        )

    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Convert DBpedia logs into CSV summaries and comparison plots.")
    parser.add_argument(
        "--hnsw-log",
        type=Path,
        default=repo_root / "hnsw/build/hnsw_dbpedia_ef_400_M_32_omp64.log",
    )
    parser.add_argument(
        "--rabitq-log",
        type=Path,
        default=repo_root / "hnsw_rabitq/logs/dbpedia/dbpedia_C_ef_400_M_32.log",
    )
    parser.add_argument("--hnsw-label", default="hnsw")
    parser.add_argument("--rabitq-label", default="rabitq_hnsw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "hnsw_rabitq/logs/dbpedia/comparison_raw_hnsw_vs_case_c",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = [
        parse_log(args.hnsw_log, args.hnsw_label),
        parse_log(args.rabitq_log, args.rabitq_label),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    apply_paper_style()
    write_csvs(datasets, args.output_dir)
    plot_summary_bars(datasets, args.output_dir)
    plot_recall_qps(datasets, args.output_dir)
    plot_ef_curves(datasets, args.output_dir)
    write_report(datasets, args.output_dir)
    print(f"Wrote comparison files to {args.output_dir}")


if __name__ == "__main__":
    main()
