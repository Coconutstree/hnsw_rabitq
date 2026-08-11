#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sched.h>
#include <x86intrin.h>

#include "../../hnswlib/rabitq_hnsw.h"
#include "../../hnswlib/space_l2.h"

namespace {

using Clock = std::chrono::steady_clock;

struct BenchmarkSample {
    double seconds = 0.0;
    double tsc_ticks = 0.0;
    double checksum = 0.0;
};

struct AsymmetricParams {
    hnswlib::RaBitQSpace *space = nullptr;
};

static float preparedAsymmetricDistance(
    const void *prepared_query,
    const void *encoded_database,
    const void *params) {
    const AsymmetricParams *typed = static_cast<const AsymmetricParams *>(params);
    return typed->space->query_distance(prepared_query, encoded_database);
}

static float oneShotAsymmetricDistance(
    const void *raw_query,
    const void *encoded_database,
    const void *params) {
    const AsymmetricParams *typed = static_cast<const AsymmetricParams *>(params);
    return typed->space->asymmetric_build_distance(raw_query, encoded_database);
}

static float buildPreparedAsymmetricDistance(
    const void *prepared_query,
    const void *encoded_database,
    const void *params) {
    const AsymmetricParams *typed = static_cast<const AsymmetricParams *>(params);
    return typed->space->asymmetric_build_distance_prepared(
        prepared_query, encoded_database);
}

static std::vector<float> readFvec(
    const std::string &path,
    size_t expected_dim,
    size_t index) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("cannot open fvec file: " + path);
    }
    const std::streamoff record_bytes = static_cast<std::streamoff>(
        sizeof(int32_t) + expected_dim * sizeof(float));
    input.seekg(static_cast<std::streamoff>(index) * record_bytes, std::ios::beg);
    int32_t stored_dim = 0;
    input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
    if (!input.good() || stored_dim != static_cast<int32_t>(expected_dim)) {
        throw std::runtime_error("fvec dimension mismatch in: " + path);
    }
    std::vector<float> result(expected_dim, 0.0f);
    input.read(
        reinterpret_cast<char *>(result.data()),
        static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input.good()) {
        throw std::runtime_error("truncated fvec record in: " + path);
    }
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, noclone))
#endif
static BenchmarkSample runKernel(
    hnswlib::DISTFUNC<float> distance,
    const void *lhs,
    const void *rhs,
    const void *param,
    uint64_t iterations) {
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    unsigned int tsc_aux = 0;
    _mm_lfence();
    const uint64_t cycles_start = __rdtscp(&tsc_aux);
    const auto start = Clock::now();

    uint64_t i = 0;
    for (; i + 4 <= iterations; i += 4) {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#endif
        sum0 += static_cast<double>(distance(lhs, rhs, param));
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#endif
        sum1 += static_cast<double>(distance(lhs, rhs, param));
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#endif
        sum2 += static_cast<double>(distance(lhs, rhs, param));
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#endif
        sum3 += static_cast<double>(distance(lhs, rhs, param));
    }
    for (; i < iterations; ++i) {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#endif
        sum0 += static_cast<double>(distance(lhs, rhs, param));
    }

    const auto end = Clock::now();
    const uint64_t cycles_end = __rdtscp(&tsc_aux);
    _mm_lfence();
    BenchmarkSample sample;
    sample.seconds = std::chrono::duration<double>(end - start).count();
    sample.tsc_ticks = static_cast<double>(cycles_end - cycles_start);
    sample.checksum = sum0 + sum1 + sum2 + sum3;
    return sample;
}

struct Kernel {
    const char *name;
    hnswlib::DISTFUNC<float> distance;
    const void *lhs;
    const void *rhs;
    const void *param;
    float one_distance;
    uint64_t iterations;

    Kernel(
        const char *kernel_name,
        hnswlib::DISTFUNC<float> kernel_distance,
        const void *kernel_lhs,
        const void *kernel_rhs,
        const void *kernel_param,
        float kernel_one_distance,
        uint64_t kernel_iterations)
        : name(kernel_name),
          distance(kernel_distance),
          lhs(kernel_lhs),
          rhs(kernel_rhs),
          param(kernel_param),
          one_distance(kernel_one_distance),
          iterations(kernel_iterations) {
    }
};

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 9) {
            std::cerr
                << "usage: distance_kernel_benchmark BASE_FVECS RABITQ_STATE"
                << " [iterations=10000000] [repetitions=5] [dimension=1536]"
                << " [lhs_id=0] [rhs_id=1] [one_shot_iterations=1000000]\n";
            return 2;
        }
        const std::string base_path = argv[1];
        const std::string state_path = argv[2];
        const uint64_t iterations = argc > 3
            ? static_cast<uint64_t>(std::strtoull(argv[3], nullptr, 10))
            : UINT64_C(10000000);
        const size_t repetitions = argc > 4
            ? static_cast<size_t>(std::strtoull(argv[4], nullptr, 10))
            : size_t{5};
        const size_t dim = argc > 5
            ? static_cast<size_t>(std::strtoull(argv[5], nullptr, 10))
            : size_t{1536};
        const size_t lhs_id = argc > 6
            ? static_cast<size_t>(std::strtoull(argv[6], nullptr, 10))
            : size_t{0};
        const size_t rhs_id = argc > 7
            ? static_cast<size_t>(std::strtoull(argv[7], nullptr, 10))
            : size_t{1};
        const uint64_t one_shot_iterations = argc > 8
            ? static_cast<uint64_t>(std::strtoull(argv[8], nullptr, 10))
            : std::min<uint64_t>(iterations, UINT64_C(1000000));
        if (iterations == 0 || repetitions == 0 || dim == 0 || one_shot_iterations == 0) {
            throw std::invalid_argument(
                "iterations, repetitions, dimension, and one-shot iterations must be positive");
        }

        const std::vector<float> raw_lhs = readFvec(base_path, dim, lhs_id);
        const std::vector<float> raw_rhs = readFvec(base_path, dim, rhs_id);

        hnswlib::RaBitQSpace::ResidualQuantizationConfig residual_config;
        residual_config.bits = hnswlib::RaBitQSpace::ResidualQuantizationBits::B4;
        residual_config.block_size = 16;
        residual_config.enabled = true;
        residual_config.enable_block_scaling = true;
        residual_config.mse_optimal_scale = true;
        residual_config.scale_fp16 = true;
        hnswlib::RaBitQSpace rabitq_space(dim, 1, 100, true, residual_config);
        {
            std::ifstream state_input(state_path, std::ios::binary);
            if (!state_input.is_open()) {
                throw std::runtime_error("cannot open RaBitQ state: " + state_path);
            }
            rabitq_space.loadState(state_input);
        }
        if (rabitq_space.get_code_layout() != hnswlib::RaBitQCodeLayout::SequentialNibble) {
            throw std::runtime_error("benchmark requires SequentialNibble layout");
        }

        const std::vector<char> encoded_lhs = rabitq_space.encodeVector(raw_lhs.data());
        const std::vector<char> encoded_rhs = rabitq_space.encodeVector(raw_rhs.data());
        const void *prepared_query = rabitq_space.prepare_query(raw_lhs.data());
        const void *build_prepared_query =
            rabitq_space.prepare_asymmetric_build_query(raw_lhs.data());

        hnswlib::L2Space float_space(dim);
        const hnswlib::DISTFUNC<float> float_distance = float_space.get_dist_func();
        const hnswlib::DISTFUNC<float> symmetric_distance = rabitq_space.get_dist_func();
        AsymmetricParams asymmetric_params;
        asymmetric_params.space = &rabitq_space;

        std::vector<Kernel> kernels;
        kernels.push_back(Kernel{
            "float32_float32",
            float_distance,
            raw_lhs.data(),
            raw_rhs.data(),
            float_space.get_dist_func_param(),
            float_distance(raw_lhs.data(), raw_rhs.data(), float_space.get_dist_func_param()),
            iterations});
        kernels.push_back(Kernel{
            "float32_primary4_prepared",
            preparedAsymmetricDistance,
            prepared_query,
            encoded_rhs.data(),
            &asymmetric_params,
            preparedAsymmetricDistance(prepared_query, encoded_rhs.data(), &asymmetric_params),
            iterations});
        kernels.push_back(Kernel{
            "float32_primary4_one_shot",
            oneShotAsymmetricDistance,
            raw_lhs.data(),
            encoded_rhs.data(),
            &asymmetric_params,
            oneShotAsymmetricDistance(raw_lhs.data(), encoded_rhs.data(), &asymmetric_params),
            one_shot_iterations});
        kernels.push_back(Kernel{
            "float32_primary4_build_prepared",
            buildPreparedAsymmetricDistance,
            build_prepared_query,
            encoded_rhs.data(),
            &asymmetric_params,
            buildPreparedAsymmetricDistance(
                build_prepared_query, encoded_rhs.data(), &asymmetric_params),
            iterations});
        kernels.push_back(Kernel{
            "primary4_primary4",
            symmetric_distance,
            encoded_lhs.data(),
            encoded_rhs.data(),
            rabitq_space.get_dist_func_param(),
            symmetric_distance(
                encoded_lhs.data(), encoded_rhs.data(), rabitq_space.get_dist_func_param()),
            iterations});

        for (const Kernel &kernel : kernels) {
            if (!std::isfinite(kernel.one_distance)) {
                throw std::runtime_error(std::string("non-finite distance for ") + kernel.name);
            }
        }

        volatile double warmup_sink = 0.0;
        for (const Kernel &kernel : kernels) {
            const uint64_t warmup_iterations = std::max<uint64_t>(
                std::min<uint64_t>(kernel.iterations / 10, 100000), 1);
            warmup_sink += runKernel(
                kernel.distance,
                kernel.lhs,
                kernel.rhs,
                kernel.param,
                warmup_iterations).checksum;
        }

        std::cout << std::setprecision(17);
        std::cout << "benchmark_schema_version=1\n";
        std::cout << "dimension=" << dim << "\n";
        std::cout << "code_dimension=" << rabitq_space.get_code_dim() << "\n";
        std::cout << "iterations_per_sample=" << iterations << "\n";
        std::cout << "one_shot_iterations_per_sample=" << one_shot_iterations << "\n";
        std::cout << "repetitions=" << repetitions << "\n";
        std::cout << "access=single_pair_hot\n";
        std::cout << "cpu=" << sched_getcpu() << "\n";
        std::cout << "lhs_id=" << lhs_id << "\n";
        std::cout << "rhs_id=" << rhs_id << "\n";
        std::cout << "float32_bytes_per_vector=" << dim * sizeof(float) << "\n";
        std::cout << "primary4_bytes_per_vector=" << rabitq_space.get_data_size() << "\n";
        std::cout << "warmup_sink=" << warmup_sink << "\n";
        std::cout << "kernel,round,iterations,seconds,ns_per_distance,tsc_ticks_per_distance,"
                     "million_distances_per_second,one_distance,checksum\n";

        volatile double final_sink = 0.0;
        for (size_t round = 0; round < repetitions; ++round) {
            for (size_t position = 0; position < kernels.size(); ++position) {
                const Kernel &kernel = kernels[(position + round) % kernels.size()];
                const BenchmarkSample sample = runKernel(
                    kernel.distance,
                    kernel.lhs,
                    kernel.rhs,
                    kernel.param,
                    kernel.iterations);
                final_sink += sample.checksum;
                std::cout << kernel.name << ',' << round << ',' << kernel.iterations << ','
                          << sample.seconds << ','
                          << sample.seconds * 1e9 / static_cast<double>(kernel.iterations) << ','
                          << sample.tsc_ticks / static_cast<double>(kernel.iterations) << ','
                          << static_cast<double>(kernel.iterations) / sample.seconds / 1e6 << ','
                          << kernel.one_distance << ',' << sample.checksum << '\n';
            }
        }
        std::cout << "final_sink=" << final_sink << "\n";
        rabitq_space.release_asymmetric_build_query(build_prepared_query);
        rabitq_space.release_query(prepared_query);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "distance_kernel_benchmark error: " << error.what() << '\n';
        return 1;
    }
}
