#include "full2bit_experiment_common.h"

#include <iomanip>

int main() {
    using namespace full2bit_exp;
    try {
        const Config cfg = load_config();
        print_config(cfg, "search_profile");
        auto index = load_full2bit_index(cfg);
        const size_t eval_queries =
            std::min(getenv_size_t("RABITQ_PROFILE_QUERIES", 100), cfg.query_count);
        const size_t topk = getenv_size_t("RABITQ_PROFILE_TOPK", 100);
        const auto efs = parse_efs(
            getenv_string("RABITQ_EFS", ""),
            std::vector<size_t>{50, 100, 150, 200, 300, 460});
        const std::vector<float> queries = read_fvecs(cfg.query_path, eval_queries, cfg.dim);

        std::cout << "full2bit_plane_bytes=" << index->space().get_full2bit_plane_bytes()
                  << " profile_payload_bytes_per_distance="
                  << index->space().profile_payload_bytes_per_distance()
                  << " hnsw_payload_bytes=" << index->space().get_data_size()
                  << " external_random_reads="
                  << (index->space().profile_uses_external_random_reads() ? 1 : 0) << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "ef\ttopK\tAvgUs\tPrepareUs\tEntryUs\tBaseUs\tDistanceUs\t"
                     "DistanceCount\tDistanceNsEach\tPayloadBytesRead\tVisited\tExpanded\t"
                     "HeapUs\tHeapPushes\tHeapPops\tResultSortUs\tExternalRandomReads\n";

        for (size_t ef : efs) {
            index->setEf(ef);
            hnswlib::HierarchicalNSW<float>::SearchProfile sum;
            for (size_t qi = 0; qi < eval_queries; ++qi) {
                hnswlib::HierarchicalNSW<float>::SearchProfile profile;
                auto result = index->index().searchKnnProfiled(
                    queries.data() + qi * cfg.dim,
                    topk,
                    profile);
                (void) result;
                sum.visited_nodes += profile.visited_nodes;
                sum.expanded_nodes += profile.expanded_nodes;
                sum.distance_computations += profile.distance_computations;
                sum.heap_pushes += profile.heap_pushes;
                sum.heap_pops += profile.heap_pops;
                sum.payload_bytes_read += profile.payload_bytes_read;
                sum.prepare_us += profile.prepare_us;
                sum.entry_us += profile.entry_us;
                sum.base_us += profile.base_us;
                sum.distance_us += profile.distance_us;
                sum.heap_us += profile.heap_us;
                sum.result_sort_us += profile.result_sort_us;
                sum.total_us += profile.total_us;
                sum.external_random_reads = sum.external_random_reads || profile.external_random_reads;
            }
            const double n = static_cast<double>(eval_queries);
            const double distance_count = static_cast<double>(sum.distance_computations) / n;
            const double distance_ns_each = sum.distance_computations == 0
                ? 0.0
                : (sum.distance_us * 1000.0) / static_cast<double>(sum.distance_computations);
            std::cout << ef << "\t" << topk << "\t"
                      << sum.total_us / n << "\t"
                      << sum.prepare_us / n << "\t"
                      << sum.entry_us / n << "\t"
                      << sum.base_us / n << "\t"
                      << sum.distance_us / n << "\t"
                      << distance_count << "\t"
                      << distance_ns_each << "\t"
                      << static_cast<double>(sum.payload_bytes_read) / n << "\t"
                      << static_cast<double>(sum.visited_nodes) / n << "\t"
                      << static_cast<double>(sum.expanded_nodes) / n << "\t"
                      << sum.heap_us / n << "\t"
                      << static_cast<double>(sum.heap_pushes) / n << "\t"
                      << static_cast<double>(sum.heap_pops) / n << "\t"
                      << sum.result_sort_us / n << "\t"
                      << (sum.external_random_reads ? 1 : 0) << "\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "full2bit_search_profile failed: " << e.what() << "\n";
        return 1;
    }
}
