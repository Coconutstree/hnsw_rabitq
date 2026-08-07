#!/usr/bin/env python3
"""Convert the dbpedia_1M NumPy/JSONL dataset to fvecs/ivecs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def write_base_fvecs(source: Path, destination: Path, batch_size: int) -> tuple[int, int]:
    vectors = np.load(source, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError(f"{source} must be a 2-D array, got shape {vectors.shape}")

    count, dimension = vectors.shape
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = np.asarray(vectors[start:end], dtype="<f4")
            records = np.empty((end - start, dimension + 1), dtype="<f4")
            records[:, 0].view("<i4")[:] = dimension
            records[:, 1:] = batch
            records.tofile(output)
            print(f"\rbase: {end:,}/{count:,}", end="", flush=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    print()
    return count, dimension


def write_queries_and_ground_truth(
    source: Path, query_destination: Path, gt_destination: Path
) -> tuple[int, int, int]:
    query_temporary = query_destination.with_suffix(query_destination.suffix + ".tmp")
    gt_temporary = gt_destination.with_suffix(gt_destination.suffix + ".tmp")
    count = 0
    dimension = None
    gt_width = None

    with (
        source.open("r", encoding="utf-8") as input_file,
        query_temporary.open("wb") as query_output,
        gt_temporary.open("wb") as gt_output,
    ):
        for line_number, line in enumerate(input_file, 1):
            item = json.loads(line)
            query = np.asarray(item["query"], dtype="<f4")
            closest_ids = np.asarray(item["closest_ids"], dtype="<i4")

            if query.ndim != 1 or closest_ids.ndim != 1:
                raise ValueError(f"line {line_number}: query and closest_ids must be arrays")
            if dimension is None:
                dimension = query.size
                gt_width = closest_ids.size
            if query.size != dimension:
                raise ValueError(
                    f"line {line_number}: dimension {query.size}, expected {dimension}"
                )
            if closest_ids.size != gt_width:
                raise ValueError(
                    f"line {line_number}: GT width {closest_ids.size}, expected {gt_width}"
                )

            np.asarray([dimension], dtype="<i4").tofile(query_output)
            query.tofile(query_output)
            np.asarray([gt_width], dtype="<i4").tofile(gt_output)
            closest_ids.tofile(gt_output)
            count += 1
            if count % 100 == 0:
                print(f"\rqueries: {count:,}", end="", flush=True)

        query_output.flush()
        gt_output.flush()
        os.fsync(query_output.fileno())
        os.fsync(gt_output.fileno())

    if count == 0 or dimension is None or gt_width is None:
        raise ValueError(f"{source} contains no records")
    query_temporary.replace(query_destination)
    gt_temporary.replace(gt_destination)
    print()
    return count, dimension, gt_width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    data_dir = args.data_dir.resolve()
    base_output = data_dir / "dbpedia_openai1536_base.fvecs"
    query_output = data_dir / "dbpedia_openai1536_query.fvecs"
    gt_output = data_dir / "dbpedia_openai1536_groundtruth.ivecs"
    outputs = (base_output, query_output, gt_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        parser.error("output already exists; use --force to replace: " + ", ".join(existing))

    base_count, base_dim = write_base_fvecs(
        data_dir / "vectors.npy", base_output, args.batch_size
    )
    query_count, query_dim, gt_width = write_queries_and_ground_truth(
        data_dir / "tests.jsonl", query_output, gt_output
    )
    if base_dim != query_dim:
        raise ValueError(f"base dimension {base_dim} != query dimension {query_dim}")

    print("conversion complete")
    print(f"base:    {base_count:,} x {base_dim} -> {base_output}")
    print(f"queries: {query_count:,} x {query_dim} -> {query_output}")
    print(f"GT:      {query_count:,} x {gt_width} -> {gt_output}")


if __name__ == "__main__":
    main()
