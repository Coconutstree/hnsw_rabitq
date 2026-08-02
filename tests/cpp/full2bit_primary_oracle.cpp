#include "full2bit_experiment_common.h"

#include <iomanip>
#include <map>

namespace {

struct ComparisonStats {
    size_t comparisons = 0;
    size_t flips = 0;
    std::vector<size_t> refine;
    std::vector<size_t> missed_flip;
    std::vector<size_t> false_refine;

    explicit ComparisonStats(size_t thresholds)
        : refine(thresholds, 0), missed_flip(thresholds, 0), false_refine(thresholds, 0) {}
};

struct BinStats {
    size_t comparisons = 0;
    size_t flips = 0;
};

struct Scored {
    hnswlib::tableint id = 0;
    float primary = 0.0f;
    float full = 0.0f;
};

bool label_to_internal_identity(
    const hnswlib::HierarchicalNSW<float> &index,
    full2bit_exp::labeltype label,
    hnswlib::tableint *internal_id) {
    return index.getInternalIdByLabelForTest(label, internal_id);
}

size_t bin_for_gap(double gap) {
    static const double edges[] = {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0};
    for (size_t i = 0; i + 1 < sizeof(edges) / sizeof(edges[0]); ++i) {
        if (gap >= edges[i] && gap < edges[i + 1]) {
            return i;
        }
    }
    return 7;
}

const char *bin_name(size_t bin) {
    static const char *names[] = {
        "[0,0.01)", "[0.01,0.02)", "[0.02,0.05)", "[0.05,0.1)",
        "[0.1,0.2)", "[0.2,0.5)", "[0.5,1.0)", "[1.0,inf)"};
    return names[std::min<size_t>(bin, 7)];
}

void add_comparison(
    std::map<std::string, ComparisonStats> &stats,
    std::map<std::string, std::vector<BinStats>> &bins,
    const std::vector<double> &thresholds,
    const std::string &type,
    double primary_gap,
    double full_gap) {
    const double norm =
        std::abs(primary_gap) / (1.0 + std::abs(primary_gap) + std::abs(full_gap));
    const bool primary_decision = primary_gap <= 0.0;
    const bool full_decision = full_gap <= 0.0;
    const bool flip = primary_decision != full_decision;

    auto found = stats.find(type);
    if (found == stats.end()) {
        found = stats.emplace(type, ComparisonStats(thresholds.size())).first;
    }
    ComparisonStats &s = found->second;
    s.comparisons++;
    s.flips += flip ? 1U : 0U;
    for (size_t i = 0; i < thresholds.size(); ++i) {
        const bool refined = norm <= thresholds[i];
        s.refine[i] += refined ? 1U : 0U;
        s.missed_flip[i] += (flip && !refined) ? 1U : 0U;
        s.false_refine[i] += (!flip && refined) ? 1U : 0U;
    }

    auto &hist = bins[type];
    if (hist.empty()) {
        hist.resize(8);
    }
    BinStats &b = hist[bin_for_gap(norm)];
    b.comparisons++;
    b.flips += flip ? 1U : 0U;
}

}  // namespace

int main() {
    using namespace full2bit_exp;
    try {
        const Config cfg = load_config();
        print_config(cfg, "primary_oracle");
        auto rabitq = load_full2bit_index(cfg);
        auto &index = rabitq->index();
        auto &space = rabitq->space();

        const size_t eval_queries =
            std::min(getenv_size_t("RABITQ_ORACLE_QUERIES", 100), cfg.query_count);
        const size_t candidate_pool = getenv_size_t("RABITQ_ORACLE_CANDIDATES", 500);
        const size_t boundary_k = getenv_size_t("RABITQ_ORACLE_BOUNDARY_K", 100);
        const size_t ef = getenv_size_t("RABITQ_ORACLE_EF", 460);
        const std::vector<double> thresholds =
            {0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0};

        const auto queries = read_fvecs(cfg.query_path, eval_queries, cfg.dim);
        rabitq->setEf(ef);

        std::map<std::string, ComparisonStats> stats;
        std::map<std::string, std::vector<BinStats>> bins;
        size_t skipped_labels = 0;
        size_t scored_candidates = 0;

        for (size_t qi = 0; qi < eval_queries; ++qi) {
            const float *query = queries.data() + qi * cfg.dim;
            const void *prepared = space.prepare_query(query);
            auto queue = index.searchKnn(query, candidate_pool);
            const auto candidates = queue_to_sorted(queue);
            std::vector<Scored> scored;
            scored.reserve(candidates.size());
            for (const auto &entry : candidates) {
                hnswlib::tableint internal_id = 0;
                if (!label_to_internal_identity(index, entry.second, &internal_id)) {
                    skipped_labels++;
                    continue;
                }
                const char *encoded = index.getDataByInternalId(internal_id);
                Scored value;
                value.id = internal_id;
                value.primary = space.query_distance_full2bit_primary_for_test(prepared, encoded);
                value.full = space.query_distance_full2bit_optimized_for_test(prepared, encoded);
                scored.push_back(value);
            }
            space.release_query(prepared);
            scored_candidates += scored.size();
            if (scored.size() < 2) {
                continue;
            }

            std::vector<Scored> by_primary = scored;
            std::sort(by_primary.begin(), by_primary.end(), [](const Scored &a, const Scored &b) {
                if (a.primary != b.primary) return a.primary < b.primary;
                return a.id < b.id;
            });
            std::vector<Scored> by_full = scored;
            std::sort(by_full.begin(), by_full.end(), [](const Scored &a, const Scored &b) {
                if (a.full != b.full) return a.full < b.full;
                return a.id < b.id;
            });

            for (size_t i = 1; i < by_primary.size(); ++i) {
                add_comparison(
                    stats,
                    bins,
                    thresholds,
                    "candidate_queue_adjacent_order",
                    static_cast<double>(by_primary[i - 1].primary) - by_primary[i].primary,
                    static_cast<double>(by_primary[i - 1].full) - by_primary[i].full);
            }

            const size_t boundary = std::min(boundary_k, by_full.size()) - 1U;
            const Scored full_worst = by_full[boundary];
            for (size_t i = boundary + 1; i < by_full.size(); ++i) {
                add_comparison(
                    stats,
                    bins,
                    thresholds,
                    "new_neighbor_vs_full_topk_worst",
                    static_cast<double>(by_full[i].primary) - full_worst.primary,
                    static_cast<double>(by_full[i].full) - full_worst.full);
            }

            if (by_primary.size() > boundary + 1U) {
                add_comparison(
                    stats,
                    bins,
                    thresholds,
                    "primary_boundary_stop_proxy",
                    static_cast<double>(by_primary[boundary].primary) - by_primary[boundary + 1U].primary,
                    static_cast<double>(by_primary[boundary].full) - by_primary[boundary + 1U].full);
            }
        }

        std::cout << std::fixed << std::setprecision(9);
        std::cout << "oracle_queries=" << eval_queries
                  << " ef=" << ef
                  << " candidate_pool=" << candidate_pool
                  << " boundary_k=" << boundary_k
                  << " scored_candidates=" << scored_candidates
                  << " skipped_missing_labels=" << skipped_labels << "\n";
        std::cout << "threshold\tcomparison_type\tcomparisons\tflip_rate\t"
                     "estimated_refinement_rate\tcritical_decision_agreement\t"
                     "flip_miss_rate\tfalse_refinement_rate\n";
        for (const auto &entry : stats) {
            const ComparisonStats &s = entry.second;
            for (size_t i = 0; i < thresholds.size(); ++i) {
                const double comparisons = static_cast<double>(std::max<size_t>(s.comparisons, 1));
                const double missed = static_cast<double>(s.missed_flip[i]);
                std::cout << thresholds[i] << "\t" << entry.first << "\t"
                          << s.comparisons << "\t"
                          << static_cast<double>(s.flips) / comparisons << "\t"
                          << static_cast<double>(s.refine[i]) / comparisons << "\t"
                          << 1.0 - missed / comparisons << "\t"
                          << missed / comparisons << "\t"
                          << static_cast<double>(s.false_refine[i]) / comparisons << "\n";
            }
        }

        std::cout << "histogram\tcomparison_type\tgap_bin\tcomparisons\tflip_rate\n";
        for (const auto &entry : bins) {
            for (size_t i = 0; i < entry.second.size(); ++i) {
                const BinStats &b = entry.second[i];
                const double comparisons = static_cast<double>(std::max<size_t>(b.comparisons, 1));
                std::cout << "histogram\t" << entry.first << "\t" << bin_name(i) << "\t"
                          << b.comparisons << "\t"
                          << static_cast<double>(b.flips) / comparisons << "\n";
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "full2bit_primary_oracle failed: " << e.what() << "\n";
        return 1;
    }
}
