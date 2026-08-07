#!/usr/bin/env python3
"""Run Graph-Turbo phase-one modes against one prebuilt RaBitQ HNSW graph."""

import argparse
import csv
import itertools
import os
import statistics
import time

import hnswlib
import numpy as np


def load_xvecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty xvecs file: {path}")
    dim = int(raw[0])
    records = raw.reshape(-1, dim + 1)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"inconsistent xvecs dimensions: {path}")
    payload = np.ascontiguousarray(records[:, 1:])
    return payload.view(dtype).reshape(-1, dim) if dtype == np.float32 else payload.astype(dtype, copy=False)


def load_array(path, dtype=None, mmap=False):
    if path.endswith(".npy"):
        value = np.load(path, mmap_mode="r" if mmap else None)
    elif path.endswith(".fvecs"):
        value = load_xvecs(path, np.float32)
    elif path.endswith(".ivecs"):
        value = load_xvecs(path, np.int32)
    else:
        raise ValueError(f"unsupported vector format: {path}")
    return np.asarray(value, dtype=dtype) if dtype is not None else value


def recall_at_k(results, truth, k):
    return sum(len(set(results[i, :k]).intersection(truth[i, :k])) for i in range(len(results))) / (len(results) * k)


def run(index, queries, truth, k, rerank_candidates, repeats):
    rows = []
    latencies = []
    labels = np.empty((len(queries), k), dtype=np.uint64)
    started = time.perf_counter()
    for repeat in range(repeats):
        for qi, query in enumerate(queries):
            q_started = time.perf_counter_ns()
            metrics = index.knn_query_with_metrics(query, k, rerank_candidates)
            latencies.append((time.perf_counter_ns() - q_started) / 1000.0)
            if repeat == repeats - 1:
                result = metrics.pop("results")
                labels[qi] = [pair[1] for pair in result]
                rows.append(metrics)
    elapsed = time.perf_counter() - started
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    samples = max(1, totals["route_oracle_samples"])
    top4 = totals["route_top4_contains_float_top1"] / samples
    top8_total = max(1, totals["route_top8_float_top4_total"])
    return {
        "recall": recall_at_k(labels, truth, k),
        "qps": repeats * len(queries) / elapsed,
        "avg_latency_us": statistics.mean(latencies),
        "p50_us": float(np.percentile(latencies, 50)),
        "p95_us": float(np.percentile(latencies, 95)),
        "p99_us": float(np.percentile(latencies, 99)),
        "route_top4_float_top1_coverage": top4,
        "route_top8_float_top4_coverage": totals["route_top8_float_top4_hits"] / top8_total,
        "total_full_distance_count": (
            totals["priority_full_count"] + totals["remaining_full_count"]
        ) / len(rows),
        **{key: value / len(rows) for key, value in totals.items()},
    }


def load_index(args):
    index = hnswlib.RaBitQIndex(
        args.dim, args.max_elements, centroid_count=1, M=args.M,
        ef_construction=args.ef_construction, random_seed=args.random_seed,
        external_residual_storage=args.external_residual_storage,
        residual_bits=args.residual_bits,
        residual_block_size=args.residual_block_size,
        residual_mse_optimal_scale=args.residual_mse_optimal_scale,
        residual_scale_fp16=args.residual_scale_fp16)
    index.load_index(args.index, args.max_elements)
    return index


def checkpoint_csv(path, rows):
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Prebuilt RaBitQ index imported from the fixed Float32 graph")
    parser.add_argument("--base", required=True, help="Float32 .npy ordered by internal ID")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--max-elements", type=int, required=True)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=100)
    parser.add_argument("--external-residual-storage", action="store_true")
    parser.add_argument("--residual-bits", type=int, default=0)
    parser.add_argument("--residual-block-size", type=int, default=16)
    parser.add_argument("--residual-mse-optimal-scale", action="store_true")
    parser.add_argument("--residual-scale-fp16", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rerank-candidates", type=int, default=100,
                        help="Maximum residual rerank candidate count; actual value is min(ef, this value)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--ef", type=int, nargs="+", default=[64, 96, 128, 192, 256, 460])
    parser.add_argument("--prefetch", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--route-bits", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--top-p", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--strategy", nargs="+", default=["equal_interval", "high_variance", "short_code_selected"])
    parser.add_argument("--matrix", choices=["full", "diagnostic"], default="full",
                        help="full runs 246 points with default lists; diagnostic keeps only equal_interval/6-bit/P=2")
    args = parser.parse_args()
    if args.rerank_candidates < args.k:
        parser.error("--rerank-candidates must be at least --k")

    base = load_array(args.base, dtype=np.float32, mmap=True)
    queries = load_array(args.queries, dtype=np.float32)
    truth = load_array(args.ground_truth)
    if base.ndim != 2 or base.shape[1] != args.dim or base.shape[0] > args.max_elements:
        raise ValueError("base vectors must be ordered by internal ID and fit max-elements/dim")

    configurations = [("baseline", 8, 0, "", 0)]
    configurations += [("batch_prefetch", pf, 0, "", 0) for pf in args.prefetch]
    if args.matrix == "full":
        configurations += [
            ("route_priority", 8, bits, strategy, top_p)
            for bits, strategy, top_p in itertools.product(args.route_bits, args.strategy, args.top_p)
        ]
    else:
        configurations.append(("route_priority", 8, 6, "equal_interval", 2))
    output_rows = []
    for mode, prefetch, bits, strategy, top_p in configurations:
        index = load_index(args)
        if mode == "route_priority":
            index.build_route_codes(base, strategy, bits)
            index.set_route_oracle(base)
        for ef in args.ef:
            index.set_ef(ef)
            index.set_graph_turbo(
                mode, top_p=max(1, top_p), prefetch_distance=prefetch,
                statistics_sample_rate=args.sample_rate if mode == "route_priority" else 0)
            rerank_candidates = min(ef, args.rerank_candidates)
            measured = run(index, queries, truth, args.k, rerank_candidates, args.repeats)
            output_rows.append({
                "dataset": args.dataset, "mode": mode, "route_bits": bits,
                "route_strategy": strategy, "top_p": top_p,
                "prefetch_distance": prefetch, "ef": ef,
                "rerank_candidates": rerank_candidates,
                "graph_fingerprint": index.graph_fingerprint(),
                "route_fingerprint": index.route_fingerprint(),
                "index_bytes": os.path.getsize(args.index), **measured,
            })
            checkpoint_csv(args.output, output_rows)
            print(mode, strategy, bits, top_p, "ef", ef,
                  "rerank", rerank_candidates,
                  "recall", measured["recall"], "qps", measured["qps"], flush=True)

    print("completed", len(output_rows), "parameter points; csv", args.output, flush=True)


if __name__ == "__main__":
    main()
