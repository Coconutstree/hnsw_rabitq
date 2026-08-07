#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <x86intrin.h>

#include "../../hnswlib/rabitq_hnsw.h"

int main(int argc, char **argv) {
    const size_t dim = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1536;
    const size_t iterations = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10000000;
    const size_t point_count = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 65536;
    if (dim == 0 || iterations == 0 || point_count == 0) {
        std::cerr << "usage: rabitq_turbo_benchmark [dimension] [iterations] [working_set_points]\n";
        return 2;
    }

    hnswlib::RaBitQHierarchicalNSW index(dim, point_count, 1, 16, 200, 100, false, false, 4);
    index.space().set_code_layout(hnswlib::RaBitQCodeLayout::SequentialNibble);
    std::vector<float> center(dim, 0.0f);
    index.space().setGlobalCenter(center.data());
    std::vector<float> query(dim, 0.0f);
    std::vector<float> point(dim, 0.0f);
    for (size_t d = 0; d < dim; ++d) {
        query[d] = std::sin(static_cast<float>(d + 1U) * 0.017f);
    }
    std::vector<std::vector<char>> encoded(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            point[d] = std::cos(static_cast<float>((i + 1U) * (d + 3U)) * 0.00013f);
        }
        encoded[i] = index.space().encodeVector(point.data());
    }

    const void *prepared = index.space().prepare_query(query.data());
    std::vector<hnswlib::PaperPruneEstimate<float>> paper_estimates(point_count);
    const size_t msb_stride = index.space().paper_msb_code_bytes();
    std::vector<uint8_t> msb_codes(point_count * msb_stride);
    std::vector<hnswlib::PaperPruneFactors<float>> paper_factors(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        paper_estimates[i] = index.space().compute_paper_prune_estimate(
            prepared, encoded[i].data(), 1.9f);
        paper_factors[i] = index.space().extract_paper_prune_sidecar(
            encoded[i].data(), msb_codes.data() + i * msb_stride);
    }
    volatile float checksum = 0.0f;
    for (size_t i = 0; i < point_count; ++i) {
        checksum += index.space().query_distance(prepared, encoded[i].data());
    }
    std::cout << "kernel,access,dimension,working_set,iterations,ns_per_distance,cycles_per_distance,checksum\n";
    const auto run = [&](const char *kernel, const char *access, size_t working_set, int kernel_kind) {
        uint64_t state = 0x9e3779b97f4a7c15ULL;
        const uint64_t cycles_start = __rdtsc();
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            size_t slot;
            if (access[0] == 'w') {
                slot = i % working_set;
            } else {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                slot = static_cast<size_t>((state >> 32) % working_set);
            }
            if (kernel_kind == 1)
                checksum += index.space().compute_short_lower_bound(prepared, encoded[slot].data());
            else if (kernel_kind == 2)
                checksum += index.space().compute_two_bit_lower_bound(prepared, encoded[slot].data());
            else if (kernel_kind == 3)
                checksum += index.space().compute_paper_prune_estimate(
                    prepared, encoded[slot].data(), 1.9f).lower_bound;
            else if (kernel_kind == 4)
                checksum += index.space().query_distance_with_paper_msb(
                    prepared, encoded[slot].data(), paper_estimates[slot].short_ip);
            else if (kernel_kind == 5) {
                const auto estimate = index.space().compute_paper_prune_estimate(
                    prepared, encoded[slot].data(), 1.9f);
                checksum += index.space().query_distance_with_paper_msb(
                    prepared, encoded[slot].data(), estimate.short_ip);
            }
            else if (kernel_kind == 6)
                checksum += index.space().compute_paper_prune_estimate_sidecar(
                    prepared, msb_codes.data() + slot * msb_stride,
                    paper_factors[slot], 1.9f).lower_bound;
            else
                checksum += index.space().query_distance(prepared, encoded[slot].data());
        }
        const auto end = std::chrono::steady_clock::now();
        const uint64_t cycles_end = __rdtsc();
        const double ns = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(
            end - start).count();
        std::cout << kernel << ',' << access << ',' << dim << ',' << working_set << ','
                  << iterations << ',' << ns / iterations << ','
                  << static_cast<double>(cycles_end - cycles_start) / iterations << ','
                  << checksum << '\n';
    };
    const size_t warm_points = std::min<size_t>(64, point_count);
    run("short_1bit", "warm", warm_points, 1);
    run("lower_bound_2bit", "warm", warm_points, 2);
    run("full_4bit", "warm", warm_points, 0);
    run("paper_msb_estimate", "warm", warm_points, 3);
    run("paper_remaining_3bit", "warm", warm_points, 4);
    run("paper_staged_full", "warm", warm_points, 5);
    run("paper_split_msb_estimate", "warm", warm_points, 6);
    run("short_1bit", "large_random", point_count, 1);
    run("lower_bound_2bit", "large_random", point_count, 2);
    run("full_4bit", "large_random", point_count, 0);
    run("paper_msb_estimate", "large_random", point_count, 3);
    run("paper_remaining_3bit", "large_random", point_count, 4);
    run("paper_staged_full", "large_random", point_count, 5);
    run("paper_split_msb_estimate", "large_random", point_count, 6);
    index.space().release_query(prepared);
    return 0;
}
