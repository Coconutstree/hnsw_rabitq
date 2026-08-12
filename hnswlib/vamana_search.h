#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hnswlib {
namespace vamana {

using NodeId = uint32_t;

struct SearchCandidate {
    float distance{0.0f};
    NodeId id{0};
    bool expanded{false};

    SearchCandidate() = default;
    SearchCandidate(float d, NodeId node, bool is_expanded)
        : distance(d), id(node), expanded(is_expanded) {}
};

struct SearchStats {
    size_t visited_nodes{0};
    size_t distance_computations{0};
    size_t hops{0};
    size_t prefetch_issued{0};
    size_t paper_checked{0};
    size_t paper_would_prune{0};
    size_t paper_not_pruned{0};
    size_t paper_full_saved{0};
    size_t paper_msb_kernel_calls{0};
    size_t paper_remaining_kernel_calls{0};
    double paper_short_time_us{0.0};
    double paper_remaining_time_us{0.0};
};

struct SearchScratch {
    std::vector<uint32_t> visited_tags;
    uint32_t current_tag{0};
    std::vector<SearchCandidate> pool;
    std::vector<NodeId> frontier;
    std::vector<NodeId> fresh;
    std::vector<float> distances;
    std::vector<PaperPruneEstimate<float>> paper_estimates;

    void reset(size_t node_count, size_t l_value, size_t beam_width, size_t start_degree) {
        if (visited_tags.size() < node_count) {
            visited_tags.resize(node_count, 0);
        }
        if (current_tag == std::numeric_limits<uint32_t>::max()) {
            std::fill(visited_tags.begin(), visited_tags.end(), 0);
            current_tag = 0;
        }
        ++current_tag;
        pool.clear();
        pool.reserve(l_value + start_degree + 1U);
        frontier.clear();
        frontier.reserve(beam_width);
        fresh.clear();
        distances.clear();
    }

    bool try_mark(NodeId id) {
        if (visited_tags[id] == current_tag) {
            return false;
        }
        visited_tags[id] = current_tag;
        return true;
    }
};

inline bool candidate_less(const SearchCandidate &lhs, const SearchCandidate &rhs) {
    return lhs.distance != rhs.distance ? lhs.distance < rhs.distance : lhs.id < rhs.id;
}

inline void insert_candidate_sorted(
    std::vector<SearchCandidate> &pool,
    SearchCandidate candidate,
    size_t l_value) {
    if (l_value == 0) {
        return;
    }
    if (pool.size() >= l_value && !candidate_less(candidate, pool.back())) {
        return;
    }
    const auto position = std::lower_bound(
        pool.begin(), pool.end(), candidate, candidate_less);
    pool.insert(position, candidate);
    if (pool.size() > l_value) {
        pool.resize(l_value);
    }
}

template <typename Graph, typename Evaluator>
const std::vector<SearchCandidate> &search_vamana(
    const Graph &graph,
    NodeId start_node,
    size_t node_count,
    size_t l_value,
    size_t beam_width,
    Evaluator &evaluator,
    SearchScratch &scratch,
    SearchStats *stats = nullptr) {
    if (node_count == 0 || l_value == 0) {
        scratch.pool.clear();
        return scratch.pool;
    }
    if (start_node >= node_count) {
        throw std::runtime_error("Vamana start node is outside graph");
    }
    if (beam_width == 0) {
        throw std::runtime_error("Vamana beam_width must be positive");
    }

    scratch.reset(node_count, l_value, beam_width, graph[start_node].size());
    std::vector<SearchCandidate> &pool = scratch.pool;

    const float start_distance = evaluator.distance(start_node);
    scratch.try_mark(start_node);
    pool.push_back(SearchCandidate{start_distance, start_node, false});
    if (stats) {
        stats->visited_nodes += 1;
        stats->distance_computations += 1;
    }

    std::vector<NodeId> &frontier = scratch.frontier;
    std::vector<NodeId> &fresh = scratch.fresh;
    std::vector<float> &distances = scratch.distances;

    while (true) {
        frontier.clear();
        for (SearchCandidate &candidate : pool) {
            if (!candidate.expanded) {
                candidate.expanded = true;
                frontier.push_back(candidate.id);
                if (frontier.size() == beam_width) {
                    break;
                }
            }
        }
        if (frontier.empty()) {
            break;
        }

        for (NodeId node : frontier) {
            fresh.clear();
            const std::vector<NodeId> &neighbors = graph[node];
            fresh.reserve(neighbors.size());
            for (NodeId neighbor : neighbors) {
                if (neighbor >= node_count) {
                    throw std::runtime_error("Vamana graph contains out-of-range neighbor");
                }
                if (scratch.try_mark(neighbor)) {
                    fresh.push_back(neighbor);
                }
            }
            if (fresh.empty()) {
                continue;
            }
            distances.resize(fresh.size());
            evaluator.distance_batch(fresh.data(), fresh.size(), distances.data());
            if (stats) {
                stats->visited_nodes += fresh.size();
                stats->distance_computations += fresh.size();
            }
            for (size_t i = 0; i < fresh.size(); ++i) {
                insert_candidate_sorted(
                    pool,
                    SearchCandidate{distances[i], fresh[i], false},
                    l_value);
            }
        }
        if (stats) {
            stats->hops += frontier.size();
        }
    }

    return pool;
}

template <typename Graph, typename Evaluator>
std::vector<SearchCandidate> search_vamana(
    const Graph &graph,
    NodeId start_node,
    size_t node_count,
    size_t l_value,
    size_t beam_width,
    Evaluator &evaluator,
    SearchStats *stats = nullptr) {
    thread_local SearchScratch scratch;
    return search_vamana(
        graph, start_node, node_count, l_value, beam_width, evaluator, scratch, stats);
}

template <typename Graph, typename Evaluator>
const std::vector<SearchCandidate> &search_vamana_paper_active(
    const Graph &graph,
    NodeId start_node,
    size_t node_count,
    size_t l_value,
    size_t beam_width,
    Evaluator &evaluator,
    float paper_epsilon0,
    size_t prefetch_distance,
    SearchScratch &scratch,
    SearchStats *stats = nullptr) {
    if (node_count == 0 || l_value == 0) {
        scratch.pool.clear();
        return scratch.pool;
    }
    if (start_node >= node_count) {
        throw std::runtime_error("Vamana start node is outside graph");
    }
    if (beam_width == 0) {
        throw std::runtime_error("Vamana beam_width must be positive");
    }

    scratch.reset(node_count, l_value, beam_width, graph[start_node].size());
    std::vector<SearchCandidate> &pool = scratch.pool;

    const float start_distance = evaluator.distance(start_node);
    scratch.try_mark(start_node);
    pool.push_back(SearchCandidate{start_distance, start_node, false});
    if (stats) {
        stats->visited_nodes += 1;
        stats->distance_computations += 1;
        stats->paper_remaining_kernel_calls += 1;
    }

    std::vector<NodeId> &frontier = scratch.frontier;
    std::vector<NodeId> &fresh = scratch.fresh;
    std::vector<PaperPruneEstimate<float>> &paper_estimates = scratch.paper_estimates;

    while (true) {
        frontier.clear();
        for (SearchCandidate &candidate : pool) {
            if (!candidate.expanded) {
                candidate.expanded = true;
                frontier.push_back(candidate.id);
                if (frontier.size() == beam_width) {
                    break;
                }
            }
        }
        if (frontier.empty()) {
            break;
        }

        for (NodeId node : frontier) {
            fresh.clear();
            const std::vector<NodeId> &neighbors = graph[node];
            fresh.reserve(neighbors.size());
            for (NodeId neighbor : neighbors) {
                if (neighbor >= node_count) {
                    throw std::runtime_error("Vamana graph contains out-of-range neighbor");
                }
                if (scratch.try_mark(neighbor)) {
                    fresh.push_back(neighbor);
                }
            }
            if (fresh.empty()) {
                continue;
            }
            if (stats) {
                stats->visited_nodes += fresh.size();
            }

            paper_estimates.assign(fresh.size(), PaperPruneEstimate<float>{});
            const auto paper_batch_start = stats
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            for (size_t i = 0; i < fresh.size(); ++i) {
                paper_estimates[i] =
                    evaluator.paper_estimate(fresh[i], paper_epsilon0);
            }
            if (stats) {
                stats->paper_short_time_us +=
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - paper_batch_start).count();
                stats->paper_msb_kernel_calls += fresh.size();
            }

            const auto score_full = [&](size_t slot) {
                const NodeId candidate_id = fresh[slot];
                const size_t pf = slot + prefetch_distance;
                if (prefetch_distance != 0 && pf < fresh.size()) {
                    evaluator.prefetch(fresh[pf]);
                    if (stats) ++stats->prefetch_issued;
                }
                const auto full_start = stats
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                const PaperPruneEstimate<float> &paper = paper_estimates[slot];
                const float distance = paper.valid
                    ? evaluator.distance_with_paper(candidate_id, paper.short_ip)
                    : evaluator.distance(candidate_id);
                if (stats) {
                    stats->paper_remaining_time_us +=
                        std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - full_start).count();
                    stats->paper_remaining_kernel_calls += 1;
                    stats->distance_computations += 1;
                }
                insert_candidate_sorted(
                    pool,
                    SearchCandidate{distance, candidate_id, false},
                    l_value);
            };

            for (size_t slot = 0; slot < fresh.size(); ++slot) {
                const NodeId candidate_id = fresh[slot];
                const bool pool_full = pool.size() >= l_value;
                const float lower_bound = pool_full
                    ? pool.back().distance
                    : std::numeric_limits<float>::max();
                const PaperPruneEstimate<float> &paper = paper_estimates[slot];
                if (pool_full && paper.valid) {
                    if (paper.valid) {
                        if (stats) {
                            stats->paper_checked += 1;
                        }
                        if (paper.lower_bound > lower_bound) {
                            if (stats) {
                                stats->paper_would_prune += 1;
                                stats->paper_full_saved += 1;
                            }
                            continue;
                        }
                        if (stats) {
                            stats->paper_not_pruned += 1;
                        }
                    }
                }
                score_full(slot);
            }
        }
        if (stats) {
            stats->hops += frontier.size();
        }
    }

    return pool;
}

template <typename Graph, typename Evaluator>
std::vector<SearchCandidate> search_vamana_paper_active(
    const Graph &graph,
    NodeId start_node,
    size_t node_count,
    size_t l_value,
    size_t beam_width,
    Evaluator &evaluator,
    float paper_epsilon0,
    size_t prefetch_distance,
    SearchStats *stats = nullptr) {
    thread_local SearchScratch scratch;
    return search_vamana_paper_active(
        graph, start_node, node_count, l_value, beam_width, evaluator,
        paper_epsilon0, prefetch_distance, scratch, stats);
}

}  // namespace vamana
}  // namespace hnswlib
