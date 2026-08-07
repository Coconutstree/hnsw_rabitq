#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def variant_name(k, layout, mode):
    return f"dbpedia_k{k}_{layout}_{mode}"


def run_variant(args, k, layout, mode):
    name = variant_name(k, layout, mode)
    for run_index in range(args.runs):
        log = args.log_dir / f"{name}_run{run_index}.log"
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(args.build_threads),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "RABITQ_DATASET": "dbpedia_openai1536",
            "RABITQ_DIM": "1536",
            "RABITQ_GT_WIDTH": "10",
            "RABITQ_BASE_PATH": str(args.data_dir / "dbpedia_openai1536_base.fvecs"),
            "RABITQ_QUERY_PATH": str(args.data_dir / "dbpedia_openai1536_query.fvecs"),
            "RABITQ_GT_PATH": str(args.data_dir / "dbpedia_openai1536_groundtruth.ivecs"),
            "RABITQ_INDEX_DIR": str(args.index_dir),
            "RABITQ_FLOAT_GRAPH_PATH": str(args.index_dir / "dbpedia_float_ef_200_M_16.bin"),
            "RABITQ_EF_CONSTRUCTION": "200",
            "RABITQ_M": "16",
            "RABITQ_CENTROID_COUNT": str(k),
            "RABITQ_CENTROID_TRAIN_SAMPLES": "100000",
            "RABITQ_CENTROID_PATH": str(args.index_dir / f"dbpedia_k{k}.centroids"),
            "RABITQ_CENTROID_MODE": mode,
            "RABITQ_CODE_LAYOUT": layout,
            "RABITQ_RESIDUAL_BITS": "4",
            "RABITQ_RERANK_CANDIDATES": "100",
            "RABITQ_STRICT_ABLATION": "1",
            "RABITQ_CSV_PATH": str(args.csv),
            "RABITQ_RUN_INDEX": str(run_index),
        })
        command = [str(args.executable)]
        if args.cpu is not None:
            command = ["taskset", "-c", str(args.cpu)] + command
        print(f"running {name} run={run_index} -> {log}", flush=True)
        with log.open("w", encoding="utf-8") as output:
            subprocess.run(command, cwd=args.repo, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True)


def main():
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run DBpedia Multi-Means/Turbo128 ablations")
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument("--data-dir", type=Path, default=repo_default / "dbpedia_1M")
    parser.add_argument("--index-dir", type=Path, default=repo_default / "build/indexes/dbpedia_ablation")
    parser.add_argument("--log-dir", type=Path, default=repo_default / "logs/dbpedia/ablation")
    parser.add_argument("--csv", type=Path, default=repo_default / "logs/dbpedia/multimeans_turbo.csv")
    parser.add_argument("--executable", type=Path, default=repo_default / "build/main")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--build-threads", type=int, default=36)
    parser.add_argument("--cpu", type=int, default=None,
                        help="pin an already-built index run to one CPU; omit during initial builds")
    parser.add_argument("--best-k", type=int, default=16)
    parser.add_argument("--phase", choices=("multimeans", "four-way"), default="multimeans")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.data_dir = args.data_dir.resolve()
    args.index_dir = args.index_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.csv = args.csv.resolve()
    args.executable = args.executable.resolve()
    if args.runs <= 0 or args.build_threads <= 0:
        parser.error("--runs and --build-threads must be positive")
    required = [
        args.executable,
        args.data_dir / "dbpedia_openai1536_base.fvecs",
        args.data_dir / "dbpedia_openai1536_query.fvecs",
        args.data_dir / "dbpedia_openai1536_groundtruth.ivecs",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing required files: " + ", ".join(missing))
    args.index_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    if args.phase == "multimeans":
        variants = [(k, "sequential", mode)
                    for k in (1, 8, 16, 32, 64, 100)
                    for mode in ("eager", "lazy")]
    else:
        variants = [
            (1, "sequential", "lazy"),
            (args.best_k, "sequential", "lazy"),
            (1, "turbo128", "lazy"),
            (args.best_k, "turbo128", "lazy"),
        ]
    for k, layout, mode in variants:
        run_variant(args, k, layout, mode)


if __name__ == "__main__":
    main()
