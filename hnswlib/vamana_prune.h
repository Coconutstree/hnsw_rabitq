#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "vamana_search.h"

namespace hnswlib {
namespace vamana {

struct PruneCandidate {
    float distance{0.0f};
    NodeId id{0};

    PruneCandidate() = default;
    PruneCandidate(float d, NodeId node) : distance(d), id(node) {}
};

inline bool prune_candidate_less(const PruneCandidate &lhs, const PruneCandidate &rhs) {
    return lhs.distance != rhs.distance ? lhs.distance < rhs.distance : lhs.id < rhs.id;
}

struct PruneScratch {
    std::vector<PruneCandidate> unique;
    std::unordered_set<NodeId> seen;
    std::vector<NodeId> selected;
    std::vector<uint8_t> disabled;
};

template <typename DistanceComputer>
std::vector<NodeId> robust_prune_4bit(
    NodeId source,
    std::vector<PruneCandidate> &candidates,
    float alpha,
    size_t degree,
    DistanceComputer &computer,
    PruneScratch *scratch = nullptr) {
    if (degree == 0) {
        return {};
    }
    if (!(alpha >= 1.0f) || !std::isfinite(alpha)) {
        throw std::runtime_error("Vamana alpha must be finite and >= 1.0");
    }

    std::sort(candidates.begin(), candidates.end(), prune_candidate_less);
    std::vector<PruneCandidate> local_unique;
    std::unordered_set<NodeId> local_seen;
    std::vector<NodeId> local_selected;
    std::vector<uint8_t> local_disabled;

    std::vector<PruneCandidate> &unique =
        scratch ? scratch->unique : local_unique;
    std::unordered_set<NodeId> &seen =
        scratch ? scratch->seen : local_seen;
    std::vector<NodeId> &selected =
        scratch ? scratch->selected : local_selected;
    std::vector<uint8_t> &disabled =
        scratch ? scratch->disabled : local_disabled;

    unique.clear();
    unique.reserve(candidates.size());
    if (candidates.size() <= 256) {
        for (const PruneCandidate &candidate : candidates) {
            if (candidate.id == source) {
                continue;
            }
            bool duplicate = false;
            for (const PruneCandidate &kept : unique) {
                if (kept.id == candidate.id) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                unique.push_back(candidate);
            }
        }
    } else {
        seen.clear();
        seen.reserve(candidates.size());
        for (const PruneCandidate &candidate : candidates) {
            if (candidate.id == source) {
                continue;
            }
            if (seen.insert(candidate.id).second) {
                unique.push_back(candidate);
            }
        }
    }

    selected.clear();
    selected.reserve(degree);
    disabled.assign(unique.size(), 0);
    float current_alpha = 1.0f;
    const float increment = std::min(alpha, 1.2f);

    while (selected.size() < degree) {
        bool promoted = false;
        for (size_t i = 0; i < unique.size() && selected.size() < degree; ++i) {
            if (disabled[i]) {
                continue;
            }
            bool occluded = false;
            for (NodeId neighbor : selected) {
                const float candidate_to_selected =
                    computer.distance_between(unique[i].id, neighbor);
                if (current_alpha * candidate_to_selected <= unique[i].distance) {
                    occluded = true;
                    break;
                }
            }
            if (occluded) {
                continue;
            }
            selected.push_back(unique[i].id);
            disabled[i] = 1;
            promoted = true;
        }
        if (current_alpha == alpha) {
            break;
        }
        const float next_alpha = std::min(alpha, current_alpha * increment);
        if (next_alpha == current_alpha) {
            break;
        }
        current_alpha = next_alpha;
        if (!promoted && selected.size() >= degree) {
            break;
        }
    }

    return std::vector<NodeId>(selected.begin(), selected.end());
}

}  // namespace vamana
}  // namespace hnswlib
