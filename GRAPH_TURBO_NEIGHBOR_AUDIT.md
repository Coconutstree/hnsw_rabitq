# Graph-Turbo neighbor-ID audit

## Current representation

- `tableint` and `linklistsizeint` are both `unsigned int`; supported builds therefore use 32-bit neighbor entries.
- Level 0 stores `linklistsizeint` followed by `maxM0_` contiguous `tableint` values inside `data_level0_memory_`.
- Upper levels store the same count-plus-entries layout in `linkLists_`, with `maxM_` entries per level.
- The deletion marker is stored in the link-list count header (`DELETE_MARK`), not in a neighbor ID. No edge-level lock, level, or deletion bits currently occupy the neighbor ID.
- The on-disk HNSW format writes the level-0 element records and upper-level link-list bytes verbatim. Graph fingerprinting previously hashed those raw neighbor bytes.

## Read and write paths

Future packed-edge work must route every entry through one codec in the following areas of `hnswalg.h`:

- Search: `searchBaseLayer`, `searchBaseLayerST`, upper-level navigation in all `searchKnn*` variants, and stop-condition/multi-vector searches.
- Construction: `mutuallyConnectNewElement`, construction `searchBaseLayer`, both `addPoint` paths, and heuristic neighbor replacement.
- Mutation: `updatePoint`, `repairConnectionsForUpdate`, `getConnectionsWithLock`, replacement of deleted nodes, deletion/un-deletion validation, and integrity checks.
- Graph lifecycle: `importGraphAndCopyDataFrom`, link-list copying, `resizeIndex`, `saveIndex`, `loadIndex`, `graphFingerprint`, debug/integrity iteration, and memory-size accounting.

`rabitq_hnsw.h` imports Float32 graph bytes through `importGraphAndCopyDataFrom`; this is also a raw-copy boundary that must be converted or validated if packed entries are introduced.

## Capacity and reserved-bit conclusion

- There are currently no reserved bits in `tableint`, so a future codec may compute `id_bits = max(1, ceil_log2(max_elements))` and `route_bits = 32 - id_bits`.
- Packed mode must be disabled when fewer than four route bits remain.
- `resizeIndex` can invalidate this split. A packed format must either freeze capacity/topology, reserve for the maximum future capacity, or atomically repack all link lists.
- This first implementation deliberately leaves all neighbor bytes and the existing index format unchanged. Route codes live in a replaceable `RouteCodeStorage` backend and an optional `.route` sidecar.

## Phase-one mutation policy

Building or loading route codes freezes insertion, update, replacement, and resize operations. Baseline and batch-only indexes retain existing mutation behavior. Deletion marking remains separate from topology and neighbor encoding.
