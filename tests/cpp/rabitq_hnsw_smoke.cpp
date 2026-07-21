#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"

int main() {
    const size_t dim = 4;
    const std::vector<float> data = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    hnswlib::RaBitQHierarchicalNSW index(dim, 4, 1, 8, 32, 0);
    index.space().setIdentityRotation();
    const size_t expected_data_size = sizeof(hnswlib::RaBitQSpace::EncodedHeader) +
                                      sizeof(hnswlib::RaBitQSpace::ShortCodeFactors) +
                                      sizeof(hnswlib::RaBitQSpace::ResidualCodeFactors) +
                                      index.space().get_compact_code_bytes() +
                                      index.space().get_residual_code_bytes();
    assert(index.space().get_data_size() == expected_data_size);

    const std::vector<char> encoded = index.space().encodeVector(data.data());
    assert(index.space().get_compact_code_bytes() ==
           (index.space().get_code_dim() * hnswlib::RaBitQSpace::kTotalBits + 7U) / 8U);
    const auto *code = reinterpret_cast<const unsigned char *>(
        encoded.data() + sizeof(hnswlib::RaBitQSpace::EncodedHeader) +
            sizeof(hnswlib::RaBitQSpace::ShortCodeFactors) +
            sizeof(hnswlib::RaBitQSpace::ResidualCodeFactors));
    const auto code_value = [code](size_t index) -> uint8_t {
        const uint8_t byte = code[index >> 1U];
        return static_cast<uint8_t>((index & 1U) ? (byte >> 4U) : (byte & 0x0FU));
    };
    for (size_t i = 0; i < index.space().get_code_dim(); ++i) {
        const uint8_t value = code_value(i);
        assert(static_cast<uint32_t>(value) <= hnswlib::RaBitQSpace::kUnsignedMax);
        const uint8_t short_bit = static_cast<uint8_t>(value >> hnswlib::RaBitQSpace::kRemainingBits);
        const uint8_t remaining = static_cast<uint8_t>(value & hnswlib::RaBitQSpace::kRemainingMax);
        assert(short_bit == 0U || short_bit == 1U);
        assert(remaining <= hnswlib::RaBitQSpace::kRemainingMax);
        const float centered = static_cast<float>(value) - hnswlib::RaBitQSpace::kUnsignedOffset;
        const float expected_centered = short_bit
                                            ? static_cast<float>(remaining) + 0.5f
                                            : -(static_cast<float>(
                                                    hnswlib::RaBitQSpace::kRemainingMax - remaining) +
                                                0.5f);
        assert(std::fabs(centered - expected_centered) < 1e-5f);
    }
    {
        float short_ip = 0.0f;
        float remaining_ip = 0.0f;
        float centered_ip = 0.0f;
        float half_sum = 0.0f;
        for (size_t i = 0; i < index.space().get_code_dim(); ++i) {
            const float q = i < dim ? data[i] : 0.0f;
            const uint8_t value = code_value(i);
            const uint8_t short_bit = static_cast<uint8_t>(value >> hnswlib::RaBitQSpace::kRemainingBits);
            const uint8_t remaining = static_cast<uint8_t>(value & hnswlib::RaBitQSpace::kRemainingMax);
            short_ip += (static_cast<float>(short_bit) - 0.5f) * q;
            remaining_ip += static_cast<float>(remaining) * q;
            centered_ip +=
                (static_cast<float>(value) - hnswlib::RaBitQSpace::kUnsignedOffset) * q;
            half_sum += q;
        }
        half_sum *= 0.5f;
        const float reconstructed =
            static_cast<float>(hnswlib::RaBitQSpace::kMsbWeight) * short_ip +
            remaining_ip -
            static_cast<float>(hnswlib::RaBitQSpace::kRemainingMax) * half_sum;
        assert(std::fabs(reconstructed - centered_ip) < 1e-5f);
    }

    const void *query_context = index.space().prepare_query(data.data());
    const float single_distance = index.space().query_distance(query_context, encoded.data());
    const hnswlib::DistanceInterval short_interval =
        index.space().compute_short_distance_interval(query_context, encoded.data());
    const hnswlib::DistanceInterval long_interval =
        index.space().compute_long_distance_interval(query_context, encoded.data());
    const hnswlib::DistanceInterval residual_interval =
        index.space().compute_residual_distance_interval(query_context, encoded.data(), long_interval.estimate);
    assert(std::isfinite(short_interval.estimate));
    assert(std::isfinite(short_interval.lower_bound));
    assert(std::isfinite(short_interval.upper_bound));
    assert(std::isfinite(long_interval.estimate));
    assert(std::isfinite(residual_interval.estimate));
    std::vector<std::vector<char>> encoded_points;
    std::vector<const void *> points;
    encoded_points.reserve(64);
    points.reserve(64);
    for (size_t i = 0; i < 64; ++i) {
        std::vector<float> shifted(data.begin(), data.begin() + dim);
        shifted[i % dim] += static_cast<float>(i) * 0.01f;
        encoded_points.push_back(index.space().encodeVector(shifted.data()));
        points.push_back(encoded_points.back().data());
    }

    for (size_t count : {size_t(1), size_t(17), size_t(31), size_t(32), size_t(33), size_t(64)}) {
        std::vector<float> batch_distance(count, 0.0f);
        index.space().batch_query_distance(query_context, points.data(), count, batch_distance.data());
        for (size_t i = 0; i < count; ++i) {
            assert(std::isfinite(batch_distance[i]));
        }
    }
    float one_distance[1] = {0.0f};
    index.space().batch_query_distance(query_context, &points[0], 1, one_distance);
    index.space().release_query(query_context);
    assert(std::fabs(single_distance - one_distance[0]) < 1e-5f);

    for (size_t i = 0; i < 4; ++i) {
        index.addPoint(data.data() + i * dim, i);
    }

    auto result = index.searchKnn(data.data(), 1);
    assert(!result.empty());
    assert(std::isfinite(result.top().first));
    hnswlib::ProgressiveSearchConfig progressive_config;
    progressive_config.efSearch = 4;
    progressive_config.max_residual_evaluations = 16;
    hnswlib::ProgressiveSearchStats progressive_stats;
    auto progressive_result =
        index.searchKnnProgressiveRefinement(data.data(), 1, progressive_config, &progressive_stats);
    assert(!progressive_result.empty());
    assert(std::isfinite(progressive_result.top().first));
    assert(progressive_stats.long_distance_evaluations > 0);
    assert(progressive_stats.residual_distance_evaluations > 0);

    const char *tmp_index = "/tmp/rabitq_hnsw_smoke.index";
    const char *tmp_state = "/tmp/rabitq_hnsw_smoke.index.rabitq";
    std::remove(tmp_index);
    std::remove(tmp_state);
    index.saveIndex(tmp_index);
    hnswlib::RaBitQHierarchicalNSW loaded(dim, 4, 1, 8, 32, 0);
    loaded.loadIndex(tmp_index, 4);
    auto loaded_result = loaded.searchKnn(data.data(), 1);
    assert(!loaded_result.empty());
    assert(loaded_result.top().second == result.top().second);
    assert(std::fabs(loaded_result.top().first - result.top().first) < 1e-5f);
    auto loaded_progressive =
        loaded.searchKnnProgressiveRefinement(data.data(), 1, progressive_config);
    assert(!loaded_progressive.empty());
    assert(std::isfinite(loaded_progressive.top().first));
    std::remove(tmp_index);
    std::remove(tmp_state);

    hnswlib::L2Space float_space(dim);
    hnswlib::HierarchicalNSW<float> float_graph(&float_space, 4, 8, 32, 0);
    for (size_t i = 0; i < 4; ++i) {
        float_graph.addPoint(data.data() + i * dim, i);
    }
    hnswlib::RaBitQHierarchicalNSW payload_build_index(dim, 4, 1, 8, 32, 0);
    payload_build_index.space().setIdentityRotation();
    const size_t payload_record_size = payload_build_index.space().get_data_size();
    std::vector<char> payloads(4 * payload_record_size, 0);
    for (size_t i = 0; i < 4; ++i) {
        payload_build_index.space().encodeVector(
            data.data() + i * dim,
            payloads.data() + i * payload_record_size);
    }
    payload_build_index.importGraphFromFloatIndexWithPayloads(
        float_graph,
        payloads,
        payload_record_size);
    assert(payload_build_index.index().getCurrentElementCount() == float_graph.getCurrentElementCount());
    assert(payload_build_index.index().data_size_ == payload_build_index.space().get_data_size());
    auto payload_build_result = payload_build_index.searchKnn(data.data(), 1);
    assert(!payload_build_result.empty());
    assert(std::isfinite(payload_build_result.top().first));

    hnswlib::RaBitQHierarchicalNSW payload_file_build_index(dim, 4, 1, 8, 32, 0);
    payload_file_build_index.space().setIdentityRotation();
    const char *tmp_payload_file = "/tmp/rabitq_hnsw_smoke.payload";
    std::remove(tmp_payload_file);
    {
        std::ofstream payload_output(tmp_payload_file, std::ios::binary);
        assert(payload_output.is_open());
        std::vector<char> encoded(payload_file_build_index.space().get_data_size(), 0);
        for (size_t i = 0; i < 4; ++i) {
            payload_file_build_index.space().encodeVector(data.data() + i * dim, encoded.data());
            payload_output.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        }
        assert(payload_output.good());
    }
    payload_file_build_index.importGraphFromFloatIndexWithPayloadFile(
        float_graph,
        tmp_payload_file,
        payload_file_build_index.space().get_data_size());
    auto payload_file_build_result = payload_file_build_index.searchKnn(data.data(), 1);
    assert(!payload_file_build_result.empty());
    assert(std::isfinite(payload_file_build_result.top().first));
    std::remove(tmp_payload_file);

    const char *tmp_floatbuild_index = "/tmp/rabitq_hnsw_smoke_floatbuild.index";
    const char *tmp_floatbuild_state = "/tmp/rabitq_hnsw_smoke_floatbuild.index.rabitq";
    std::remove(tmp_floatbuild_index);
    std::remove(tmp_floatbuild_state);
    payload_build_index.saveIndex(tmp_floatbuild_index);
    hnswlib::RaBitQHierarchicalNSW loaded_floatbuild(dim, 4, 1, 8, 32, 0);
    loaded_floatbuild.loadIndex(tmp_floatbuild_index, 4);
    assert(loaded_floatbuild.index().data_size_ == loaded_floatbuild.space().get_data_size());
    auto loaded_floatbuild_result = loaded_floatbuild.searchKnn(data.data(), 1);
    assert(!loaded_floatbuild_result.empty());
    assert(loaded_floatbuild_result.top().second == payload_build_result.top().second);
    std::remove(tmp_floatbuild_index);
    std::remove(tmp_floatbuild_state);

    std::cout << "RaBitQ HNSW smoke test passed\n";
    return 0;
}
