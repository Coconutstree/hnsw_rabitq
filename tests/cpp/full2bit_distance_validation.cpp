#include "full2bit_experiment_common.h"

#include <iomanip>
#include <random>

int main() {
    using namespace full2bit_exp;
    try {
        const Config cfg = load_config();
        print_config(cfg, "distance_validation");
        auto index = load_full2bit_index(cfg);
        const size_t validation_queries =
            std::min(getenv_size_t("RABITQ_VALIDATE_QUERIES", 50), cfg.query_count);
        const size_t candidates_per_query = getenv_size_t("RABITQ_VALIDATE_CANDIDATES", 100);
        const std::vector<float> queries = read_fvecs(cfg.query_path, validation_queries, cfg.dim);

        std::cout << "full2bit_plane_bytes=" << index->space().get_full2bit_plane_bytes()
                  << " expected_code_plane_bytes_per_distance="
                  << 2U * index->space().get_full2bit_plane_bytes()
                  << " hnsw_payload_bytes=" << index->space().get_data_size() << "\n";

        std::mt19937 rng(100);
        std::uniform_int_distribution<size_t> id_dist(
            0, static_cast<size_t>(index->index().cur_element_count) - 1U);

        double direct_split_abs = 0.0;
        double direct_opt_abs = 0.0;
        double split_opt_abs = 0.0;
        double max_direct_split_abs = 0.0;
        double max_direct_opt_abs = 0.0;
        size_t pair_count = 0;
        size_t u_mismatch = 0;
        double top10_overlap_sum = 0.0;
        double top100_overlap_sum = 0.0;
        double exact_order_match = 0.0;

        for (size_t qi = 0; qi < validation_queries; ++qi) {
            const void *prepared = index->space().prepare_query(queries.data() + qi * cfg.dim);
            std::vector<std::pair<float, hnswlib::tableint>> direct;
            std::vector<std::pair<float, hnswlib::tableint>> split;
            std::vector<std::pair<float, hnswlib::tableint>> opt;
            direct.reserve(candidates_per_query);
            split.reserve(candidates_per_query);
            opt.reserve(candidates_per_query);

            for (size_t ci = 0; ci < candidates_per_query; ++ci) {
                const hnswlib::tableint internal_id =
                    static_cast<hnswlib::tableint>(id_dist(rng));
                const char *encoded = index->index().getDataByInternalId(internal_id);
                for (size_t d = 0; d < cfg.dim; ++d) {
                    const uint8_t p = index->space().full2bit_primary_bit_for_test(encoded, d);
                    const uint8_t r = index->space().full2bit_residual_bit_for_test(encoded, d);
                    const uint8_t u = index->space().full2bit_code_value_for_test(encoded, d);
                    if (u != static_cast<uint8_t>(2U * p + r)) {
                        u_mismatch++;
                    }
                }

                const float d_direct =
                    index->space().query_distance_full2bit_reference_for_test(prepared, encoded);
                const float d_split =
                    index->space().query_distance_full2bit_split_scalar_for_test(prepared, encoded);
                const float d_opt =
                    index->space().query_distance_full2bit_optimized_for_test(prepared, encoded);
                const double ds = std::abs(static_cast<double>(d_direct) - d_split);
                const double do_abs = std::abs(static_cast<double>(d_direct) - d_opt);
                const double so = std::abs(static_cast<double>(d_split) - d_opt);
                direct_split_abs += ds;
                direct_opt_abs += do_abs;
                split_opt_abs += so;
                max_direct_split_abs = std::max(max_direct_split_abs, ds);
                max_direct_opt_abs = std::max(max_direct_opt_abs, do_abs);
                pair_count++;
                direct.emplace_back(d_direct, internal_id);
                split.emplace_back(d_split, internal_id);
                opt.emplace_back(d_opt, internal_id);
            }
            index->space().release_query(prepared);

            typedef std::pair<float, hnswlib::tableint> ScoredInternal;
            struct Sorter {
                bool operator()(const ScoredInternal &a, const ScoredInternal &b) const {
                if (a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
                }
            } sorter;
            std::sort(direct.begin(), direct.end(), sorter);
            std::sort(split.begin(), split.end(), sorter);
            std::sort(opt.begin(), opt.end(), sorter);
            size_t same_order = 0;
            for (size_t i = 0; i < direct.size(); ++i) {
                same_order += direct[i].second == opt[i].second ? 1U : 0U;
            }
            exact_order_match += direct.empty()
                ? 0.0
                : static_cast<double>(same_order) / static_cast<double>(direct.size());

            struct Overlap {
                double operator()(
                    const std::vector<ScoredInternal> &lhs,
                    const std::vector<ScoredInternal> &rhs,
                    size_t k) const {
                std::unordered_set<hnswlib::tableint> expected;
                const size_t limit = std::min(k, lhs.size());
                for (size_t i = 0; i < limit; ++i) {
                    expected.insert(lhs[i].second);
                }
                size_t hits = 0;
                for (size_t i = 0; i < std::min(k, rhs.size()); ++i) {
                    hits += expected.count(rhs[i].second) != 0 ? 1U : 0U;
                }
                return limit == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(limit);
                }
            } overlap;
            top10_overlap_sum += overlap(direct, opt, 10);
            top100_overlap_sum += overlap(direct, opt, 100);
        }

        std::cout << std::fixed << std::setprecision(9);
        std::cout << "pairs=" << pair_count
                  << " queries=" << validation_queries
                  << " candidates_per_query=" << candidates_per_query << "\n";
        std::cout << "u_equals_2p_plus_r_mismatches=" << u_mismatch << "\n";
        std::cout << "direct_vs_split_mae=" << direct_split_abs / pair_count
                  << " max_abs=" << max_direct_split_abs << "\n";
        std::cout << "direct_vs_optimized_mae=" << direct_opt_abs / pair_count
                  << " max_abs=" << max_direct_opt_abs << "\n";
        std::cout << "split_vs_optimized_mae=" << split_opt_abs / pair_count << "\n";
        std::cout << "optimized_exact_order_match="
                  << exact_order_match / static_cast<double>(validation_queries) << "\n";
        std::cout << "optimized_top10_overlap_with_direct="
                  << top10_overlap_sum / static_cast<double>(validation_queries) << "\n";
        std::cout << "optimized_top100_overlap_with_direct="
                  << top100_overlap_sum / static_cast<double>(validation_queries) << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "full2bit_distance_validation failed: " << e.what() << "\n";
        return 1;
    }
}
