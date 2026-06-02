#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


BUILD_PROGRESS_RE = re.compile(
    r"^\s*(?P<progress>\d+(?:\.\d+)?)\s*%,\s*"
    r"(?P<kips>\d+(?:\.\d+)?)\s*kips\s+Mem:\s*"
    r"(?P<memory>\d+(?:\.\d+)?)\s*Mb"
)
BUILD_TIME_RE = re.compile(r"^Build time:(?P<seconds>\d+(?:\.\d+)?)\s+seconds")
EVAL_PLAIN_RE = re.compile(
    r"^\s*(?P<ef>\d+)\s+(?P<recall>\d+(?:\.\d+)?)\s+(?P<us>\d+(?:\.\d+)?)\s+us\s*$"
)
EVAL_KEYVALUE_RE = re.compile(
    r"^\s*ef=(?P<ef>\d+)\s+recall=(?P<recall>\d+(?:\.\d+)?)\s+us=(?P<us>\d+(?:\.\d+)?)\s*$"
)
MEMORY_USAGE_RE = re.compile(r"^Actual memory usage:\s*(?P<memory>\d+(?:\.\d+)?)\s*Mb")


def parse_log(log_path: Path, label: str) -> dict:
    build_progress = []
    eval_points = []
    build_time_seconds = None
    final_memory_mb = None

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = BUILD_PROGRESS_RE.match(line)
        if match:
            build_progress.append(
                {
                    "progress_pct": float(match.group("progress")),
                    "kips": float(match.group("kips")),
                    "memory_mb": float(match.group("memory")),
                }
            )
            continue

        match = BUILD_TIME_RE.match(line)
        if match:
            build_time_seconds = float(match.group("seconds"))
            continue

        match = MEMORY_USAGE_RE.match(line)
        if match:
            final_memory_mb = float(match.group("memory"))
            continue

        match = EVAL_KEYVALUE_RE.match(line) or EVAL_PLAIN_RE.match(line)
        if match:
            eval_points.append(
                {
                    "ef": int(match.group("ef")),
                    "recall": float(match.group("recall")),
                    "us": float(match.group("us")),
                }
            )

    if not build_progress:
        raise ValueError(f"No build progress lines found in {log_path}")
    if not eval_points:
        raise ValueError(f"No evaluation lines found in {log_path}")
    if build_time_seconds is None:
        raise ValueError(f"No build time found in {log_path}")
    if final_memory_mb is None:
        raise ValueError(f"No final memory usage found in {log_path}")

    return {
        "label": label,
        "path": log_path,
        "build_progress": build_progress,
        "eval_points": eval_points,
        "build_time_seconds": build_time_seconds,
        "final_memory_mb": final_memory_mb,
        "peak_build_memory_mb": max(point["memory_mb"] for point in build_progress),
        "avg_build_kips": sum(point["kips"] for point in build_progress) / len(build_progress),
    }


def plot_recall_vs_latency(ax, datasets: list[dict]) -> None:
    for dataset in datasets:
        xs = [point["us"] for point in dataset["eval_points"]]
        ys = [point["recall"] for point in dataset["eval_points"]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=dataset["label"])
    ax.set_xlabel("Latency per query (us)")
    ax.set_ylabel("Recall@1")
    ax.set_title("Recall vs Latency")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_recall_vs_ef(ax, datasets: list[dict]) -> None:
    for dataset in datasets:
        xs = [point["ef"] for point in dataset["eval_points"]]
        ys = [point["recall"] for point in dataset["eval_points"]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=dataset["label"])
    ax.set_xlabel("ef")
    ax.set_ylabel("Recall@1")
    ax.set_title("Recall vs ef")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_latency_vs_ef(ax, datasets: list[dict]) -> None:
    for dataset in datasets:
        xs = [point["ef"] for point in dataset["eval_points"]]
        ys = [point["us"] for point in dataset["eval_points"]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=dataset["label"])
    ax.set_xlabel("ef")
    ax.set_ylabel("Latency per query (us)")
    ax.set_title("Latency vs ef")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_build_kips(ax, datasets: list[dict]) -> None:
    for dataset in datasets:
        xs = [point["progress_pct"] for point in dataset["build_progress"]]
        ys = [point["kips"] for point in dataset["build_progress"]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=dataset["label"])
    ax.set_xlabel("Build progress (%)")
    ax.set_ylabel("Build speed (kips)")
    ax.set_title("Build Speed vs Progress")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_build_memory(ax, datasets: list[dict]) -> None:
    for dataset in datasets:
        xs = [point["progress_pct"] for point in dataset["build_progress"]]
        ys = [point["memory_mb"] / 1024.0 for point in dataset["build_progress"]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=dataset["label"])
    ax.set_xlabel("Build progress (%)")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Build Memory vs Progress")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_summary_bars(axs, datasets: list[dict]) -> None:
    labels = [dataset["label"] for dataset in datasets]
    x_positions = list(range(len(datasets)))

    build_times = [dataset["build_time_seconds"] for dataset in datasets]
    peak_memories_gb = [dataset["peak_build_memory_mb"] / 1024.0 for dataset in datasets]
    final_memories_gb = [dataset["final_memory_mb"] / 1024.0 for dataset in datasets]
    avg_build_kips = [dataset["avg_build_kips"] for dataset in datasets]

    summaries = [
        ("Build Time (s)", build_times),
        ("Peak Build Memory (GB)", peak_memories_gb),
        ("Final Memory (GB)", final_memories_gb),
        ("Average Build Speed (kips)", avg_build_kips),
    ]

    for ax, (title, values) in zip(axs, summaries):
        bars = ax.bar(x_positions, values, width=0.55)
        ax.set_xticks(x_positions, labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def save_individual_figures(datasets: list[dict], output_dir: Path) -> None:
    figures = [
        ("recall_vs_latency.png", plot_recall_vs_latency),
        ("recall_vs_ef.png", plot_recall_vs_ef),
        ("latency_vs_ef.png", plot_latency_vs_ef),
        ("build_kips_vs_progress.png", plot_build_kips),
        ("build_memory_vs_progress.png", plot_build_memory),
    ]

    for filename, plotter in figures:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotter(ax, datasets)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_summary_bars(axs.flatten(), datasets)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_bars.png", dpi=200)
    plt.close(fig)


def save_overview_figure(datasets: list[dict], output_dir: Path) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    plot_recall_vs_latency(axs[0][0], datasets)
    plot_recall_vs_ef(axs[0][1], datasets)
    plot_latency_vs_ef(axs[1][0], datasets)
    plot_build_kips(axs[1][1], datasets)
    plot_build_memory(axs[2][0], datasets)
    summary_ax = axs[2][1]
    labels = [dataset["label"] for dataset in datasets]
    summary_lines = []
    for dataset in datasets:
        best_point = max(dataset["eval_points"], key=lambda point: point["recall"])
        summary_lines.append(
            f"{dataset['label']}\n"
            f"build={dataset['build_time_seconds']:.2f}s, "
            f"peak_mem={dataset['peak_build_memory_mb'] / 1024.0:.2f}GB, "
            f"best_recall={best_point['recall']:.4f} @ ef={best_point['ef']}, "
            f"latency={best_point['us']:.2f}us"
        )
    summary_ax.axis("off")
    summary_ax.set_title("Quick Summary")
    summary_ax.text(
        0.02,
        0.98,
        "\n\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
    )

    fig.suptitle("BigANN Performance Comparison: hnsw vs hnsw_rabitq", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "comparison_overview.png", dpi=200)
    plt.close(fig)


def write_summary_text(datasets: list[dict], output_dir: Path) -> None:
    lines = []
    for dataset in datasets:
        best_recall = max(dataset["eval_points"], key=lambda point: point["recall"])
        best_latency = min(dataset["eval_points"], key=lambda point: point["us"])
        lines.append(f"[{dataset['label']}]")
        lines.append(f"log={dataset['path']}")
        lines.append(f"build_time_seconds={dataset['build_time_seconds']:.4f}")
        lines.append(f"peak_build_memory_mb={dataset['peak_build_memory_mb']:.2f}")
        lines.append(f"final_memory_mb={dataset['final_memory_mb']:.2f}")
        lines.append(f"avg_build_kips={dataset['avg_build_kips']:.4f}")
        lines.append(
            "best_recall="
            f"{best_recall['recall']:.4f} at ef={best_recall['ef']} latency_us={best_recall['us']:.4f}"
        )
        lines.append(
            "fastest_query="
            f"{best_latency['us']:.4f}us at ef={best_latency['ef']} recall={best_latency['recall']:.4f}"
        )
        lines.append("")

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "build"
    parser = argparse.ArgumentParser(
        description="Plot performance comparisons for hnsw and hnsw_rabitq BigANN logs."
    )
    parser.add_argument(
        "--hnsw-log",
        type=Path,
        default=build_dir / "test_hnsw.log",
        help="Path to the original hnsw log file.",
    )
    parser.add_argument(
        "--rabitq-log",
        type=Path,
        default=build_dir / "test_rabitq_hnsw.log",
        help="Path to the modified hnsw_rabitq log file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=build_dir / "comparison_results",
        help="Directory where the figures will be saved.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    datasets = [
        parse_log(args.hnsw_log, "hnsw"),
        parse_log(args.rabitq_log, "hnsw_rabitq"),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    save_individual_figures(datasets, args.output_dir)
    save_overview_figure(datasets, args.output_dir)
    write_summary_text(datasets, args.output_dir)

    print(f"Saved comparison plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
