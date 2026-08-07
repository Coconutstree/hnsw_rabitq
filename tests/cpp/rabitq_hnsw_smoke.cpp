#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"

static void test_residual_pack_roundtrip() {
    const std::vector<size_t> bits_list = {1, 2, 4, 8, 16};
    const std::vector<size_t> dims = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 128, 1536};
    for (const size_t bits : bits_list) {
        const int32_t qmax = hnswlib::RaBitQSpace::residual_quantized_max(bits);
        for (const size_t dim : dims) {
            std::vector<int32_t> codes(dim, 0);
            for (size_t i = 0; i < dim; ++i) {
                if (bits == 1) {
                    codes[i] = (i & 1U) ? 1 : -1;
                } else {
                    const int32_t span = 2 * qmax + 1;
                    codes[i] = static_cast<int32_t>(i % static_cast<size_t>(span)) - qmax;
                    if (bits == 2 && codes[i] == -2) {
                        codes[i] = -1;
                    }
                }
            }
            const size_t bytes = hnswlib::RaBitQSpace::residual_packed_code_size_bytes(dim, bits);
            std::vector<uint8_t> packed(bytes, 0xFF);
            std::vector<int32_t> decoded(dim, 0);
            hnswlib::RaBitQSpace::pack_residual_codes(codes.data(), dim, bits, packed.data());
            hnswlib::RaBitQSpace::unpack_residual_codes(packed.data(), dim, bits, decoded.data());
            assert(decoded == codes);
            if ((dim * bits) % 8U != 0U) {
                const uint8_t used = static_cast<uint8_t>((dim * bits) & 7U);
                const uint8_t unused_mask = static_cast<uint8_t>(0xFFU << used);
                assert((packed.back() & unused_mask) == 0U);
            }
        }
    }
}

static void run_residual_bits_smoke(size_t residual_bits) {
    const size_t dim = 4;
    const std::vector<float> data = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    hnswlib::RaBitQHierarchicalNSW index(dim, 4, 1, 8, 32, 0, false, false, residual_bits);
    index.space().setIdentityRotation();
    assert(index.space().get_residual_bits() == residual_bits);
    assert(index.space().get_compact_code_bytes() == (index.space().get_code_dim() + 1U) / 2U);
    assert(index.space().get_residual_code_bytes() ==
           hnswlib::RaBitQSpace::residual_packed_code_size_bytes(index.space().get_code_dim(), residual_bits));
    for (size_t i = 0; i < 4; ++i) {
        index.addPoint(data.data() + i * dim, i);
    }
    auto result = index.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!result.empty());
    assert(std::isfinite(result.top().first));

    const std::string suffix = std::to_string(residual_bits);
    const std::string tmp_index = "/tmp/rabitq_hnsw_smoke_r" + suffix + ".index";
    const std::string tmp_state = tmp_index + ".rabitq";
    std::remove(tmp_index.c_str());
    std::remove(tmp_state.c_str());
    index.saveIndex(tmp_index);
    hnswlib::RaBitQHierarchicalNSW loaded(dim, 4, 1, 8, 32, 0, false, false, residual_bits);
    loaded.loadIndex(tmp_index, 4);
    auto loaded_result = loaded.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!loaded_result.empty());
    assert(std::isfinite(loaded_result.top().first));
    std::remove(tmp_index.c_str());
    std::remove(tmp_state.c_str());

    hnswlib::L2Space float_space(dim);
    hnswlib::HierarchicalNSW<float> float_graph(&float_space, 4, 8, 32, 0);
    for (size_t i = 0; i < 4; ++i) {
        float_graph.addPoint(data.data() + i * dim, i);
    }
    hnswlib::RaBitQHierarchicalNSW external_index(dim, 4, 1, 8, 32, 0, false, true, residual_bits);
    external_index.space().setIdentityRotation();
    const std::string tmp_external_residual = "/tmp/rabitq_hnsw_smoke_external_r" + suffix + ".residual";
    const std::string tmp_external_payload = "/tmp/rabitq_hnsw_smoke_external_r" + suffix + ".payload";
    std::remove(tmp_external_residual.c_str());
    std::remove(tmp_external_payload.c_str());
    {
        std::ofstream payload_output(tmp_external_payload, std::ios::binary);
        assert(payload_output.is_open());
        std::vector<char> full_encoded(external_index.space().get_full_data_size(), 0);
        for (size_t i = 0; i < 4; ++i) {
            external_index.space().encodeVectorFull(data.data() + i * dim, full_encoded.data());
            payload_output.write(full_encoded.data(), static_cast<std::streamsize>(full_encoded.size()));
        }
        assert(payload_output.good());
    }
    external_index.importGraphFromFloatIndexWithFullPayloadFileAndExternalResiduals(
        float_graph,
        tmp_external_payload,
        tmp_external_residual,
        external_index.space().get_full_data_size());
    auto external_result =
        external_index.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!external_result.empty());
    assert(std::isfinite(external_result.top().first));
    const char *tmp_external_bfs_residual =
        "/tmp/rabitq_hnsw_smoke_external_bfs.index.residual";
    std::remove(tmp_external_bfs_residual);
    hnswlib::RaBitQHierarchicalNSW external_bfs(
        dim, 4, 1, 8, 32, 0, false, true, residual_bits);
    external_bfs.importBfsReorderedFrom(external_index, tmp_external_bfs_residual);
    assert(external_bfs.labelGraphFingerprint() == external_index.labelGraphFingerprint());
    assert(external_bfs.payloadFingerprintByLabel() == external_index.payloadFingerprintByLabel());
    assert(external_bfs.residualFingerprintByLabel() == external_index.residualFingerprintByLabel());
    auto external_bfs_result =
        external_bfs.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!external_bfs_result.empty());
    assert(external_bfs_result.top().second == external_result.top().second);
    std::remove(tmp_external_residual.c_str());
    std::remove(tmp_external_payload.c_str());
    std::remove(tmp_external_bfs_residual);
}

static void test_multimeans_eager_lazy_equivalence() {
    const size_t dim = 4;
    const std::vector<float> centroids = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::vector<float> points = {
        1.0f, 0.1f, 0.0f, 0.0f,
        0.9f, 0.0f, 0.1f, 0.0f,
        0.0f, 1.0f, 0.1f, 0.0f,
        0.1f, 0.9f, 0.0f, 0.1f,
    };
    hnswlib::RaBitQHierarchicalNSW index(dim, 4, 2, 8, 32, 0, false, false, 4);
    index.space().setIdentityRotation();
    index.space().setCentroids(centroids.data(), 2);
    for (size_t i = 0; i < 4; ++i) {
        index.addPoint(points.data() + i * dim, i);
    }

    hnswlib::RaBitQSearchMetrics eager_metrics;
    index.space().set_centroid_query_mode(hnswlib::RaBitQSpace::CentroidQueryMode::Eager);
    auto eager = index.searchKnnPlainThenResidualRerank(points.data(), 2, 4, nullptr, &eager_metrics);
    hnswlib::RaBitQSearchMetrics lazy_metrics;
    index.space().set_centroid_query_mode(hnswlib::RaBitQSpace::CentroidQueryMode::Lazy);
    auto lazy = index.searchKnnPlainThenResidualRerank(points.data(), 2, 4, nullptr, &lazy_metrics);
    assert(eager_metrics.active_centroids == 2);
    assert(lazy_metrics.active_centroids > 0 && lazy_metrics.active_centroids <= 2);
    assert(eager_metrics.distance_computations == lazy_metrics.distance_computations);
    assert(eager_metrics.visited_nodes == lazy_metrics.visited_nodes);
    while (!eager.empty()) {
        assert(!lazy.empty());
        assert(eager.top().second == lazy.top().second);
        assert(std::fabs(eager.top().first - lazy.top().first) < 1e-5f);
        eager.pop();
        lazy.pop();
    }
    assert(lazy.empty());

    const std::string centroid_path = "/tmp/rabitq_multimeans.centroids";
    {
        std::ofstream output(centroid_path, std::ios::binary);
        index.space().saveCentroids(output, 123U, 1000U);
    }
    hnswlib::RaBitQHierarchicalNSW restored(dim, 4, 2, 8, 32, 0, false, false, 4);
    uint32_t restored_seed = 0;
    uint64_t restored_samples = 0;
    {
        std::ifstream input(centroid_path, std::ios::binary);
        restored.space().loadCentroids(input, &restored_seed, &restored_samples);
    }
    assert(restored_seed == 123U);
    assert(restored_samples == 1000U);
    for (size_t i = 0; i < 4; ++i) {
        assert(restored.space().assignCentroid(points.data() + i * dim) ==
               index.space().assignCentroid(points.data() + i * dim));
    }
    std::remove(centroid_path.c_str());
}

static void test_asymmetric_four_bit_construction() {
    constexpr size_t dim = 4;
    constexpr size_t count = 8;
    const std::vector<float> data = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
        1,1,0,0, 0,1,1,0, 0,0,1,1, 1,0,0,1,
    };
    hnswlib::RaBitQHierarchicalNSW index(dim, count, 1, 2, 8, 100);
    index.space().setIdentityRotation();
    const std::vector<char> encoded_reference = index.space().encodeVector(data.data() + dim);
    const void *prepared = index.space().prepare_query(data.data());
    const float expected = index.space().query_distance(prepared, encoded_reference.data());
    const float actual = index.space().asymmetric_build_distance(
        data.data(), encoded_reference.data());
    index.space().release_query(prepared);
    assert(std::fabs(expected - actual) < 1e-6f);
    index.setAsymmetricBuildRawProvider(
        [&data](hnswlib::labeltype label) -> const void * {
            return data.data() + static_cast<size_t>(label) * dim;
        });
    for (size_t label = 0; label < count; ++label)
        index.addPointAsymmetric(data.data() + label * dim, label);
    assert(index.asymmetricBuildDistanceCalls() > 0);
    assert(index.encodedBuildDistanceCalls() == 0);
    index.clearAsymmetricBuildRawProvider();
    auto result = index.searchKnn(data.data(), 1);
    assert(!result.empty());
    assert(result.top().second == 0);

    const std::string payload_path = "/tmp/rabitq_asym_external.payload";
    const std::string residual_path = "/tmp/rabitq_asym_external.residual";
    std::remove(payload_path.c_str());
    std::remove(residual_path.c_str());
    hnswlib::RaBitQHierarchicalNSW external(dim, count, 1, 2, 8, 100, false, true, 4);
    external.space().setIdentityRotation();
    external.setAsymmetricBuildRawProvider(
        [&data](hnswlib::labeltype label) -> const void * {
            return data.data() + static_cast<size_t>(label) * dim;
        });
    std::ofstream payload(payload_path, std::ios::binary);
    assert(payload.is_open());
    for (size_t label = 0; label < count; ++label) {
        std::vector<char> full(external.space().get_full_data_size(), 0);
        std::vector<char> compact(external.space().get_data_size(), 0);
        external.space().encodeVectorFull(data.data() + label * dim, full.data());
        external.space().copyCompactPayloadFromFull(full.data(), compact.data());
        payload.write(full.data(), static_cast<std::streamsize>(full.size()));
        external.addPointAsymmetric(data.data() + label * dim, label, compact.data());
    }
    payload.close();
    external.materializeExternalResidualsFromFullPayloadFile(
        payload_path, residual_path, external.space().get_full_data_size());
    assert(external.residualFingerprintByLabel() != 0);
    external.clearAsymmetricBuildRawProvider();
    std::remove(payload_path.c_str());
    std::remove(residual_path.c_str());
}

static void test_symmetric_four_bit_construction() {
    constexpr size_t dim = 4;
    constexpr size_t count = 8;
    const std::vector<float> data = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
        1,1,0,0, 0,1,1,0, 0,0,1,1, 1,0,0,1,
    };
    hnswlib::RaBitQHierarchicalNSW index(dim, count, 1, 2, 8, 100);
    index.space().setIdentityRotation();

    const std::vector<char> lhs = index.space().encodeVector(data.data());
    const std::vector<char> rhs = index.space().encodeVector(data.data() + dim);
    const auto distance = index.space().get_dist_func();
    const void *distance_param = index.space().get_dist_func_param();
    const float self_distance = distance(lhs.data(), lhs.data(), distance_param);
    const float lhs_rhs = distance(lhs.data(), rhs.data(), distance_param);
    const float rhs_lhs = distance(rhs.data(), lhs.data(), distance_param);
    assert(std::fabs(self_distance) < 1e-6f);
    assert(std::isfinite(lhs_rhs));
    assert(lhs_rhs >= 0.0f && lhs_rhs <= 4.0f);
    assert(std::fabs(lhs_rhs - rhs_lhs) < 1e-6f);

    for (size_t label = 0; label < count; ++label)
        index.addPoint(data.data() + label * dim, label);
    assert(index.asymmetricBuildDistanceCalls() == 0);
    assert(index.encodedBuildDistanceCalls() > 0);
    auto result = index.searchKnn(data.data(), 1);
    assert(!result.empty());
    assert(result.top().second == 0);
}

static void test_turbo128_layout() {
    for (const size_t dim : {size_t(128), size_t(256), size_t(960), size_t(1536)}) {
        size_t code_dim = 1;
        const size_t rounded_dim = ((dim + 63U) / 64U) * 64U;
        while (code_dim < rounded_dim) code_dim <<= 1U;
        std::vector<uint8_t> values(code_dim, 0);
        std::vector<uint8_t> sequential(code_dim / 2U, 0);
        std::vector<uint8_t> turbo(code_dim / 2U, 0);
        std::vector<uint8_t> unpacked(code_dim, 0);
        for (size_t i = 0; i < code_dim; ++i) {
            values[i] = static_cast<uint8_t>((i * 13U + 7U) & 0x0FU);
            if (i & 1U) sequential[i >> 1U] |= static_cast<uint8_t>(values[i] << 4U);
            else sequential[i >> 1U] |= values[i];
        }
        hnswlib::RaBitQSpace::convertSequentialToTurbo(
            sequential.data(), turbo.data(), code_dim);
        hnswlib::RaBitQSpace::unpackTurbo128(turbo.data(), unpacked.data(), code_dim);
        assert(values == unpacked);

        std::vector<float> point(dim, 0.0f);
        std::vector<float> query(dim, 0.0f);
        std::vector<float> center(dim, 0.0f);
        for (size_t i = 0; i < dim; ++i) {
            point[i] = std::sin(static_cast<float>(i + 1U) * 0.013f);
            query[i] = std::cos(static_cast<float>(i + 3U) * 0.017f);
        }
        hnswlib::RaBitQHierarchicalNSW seq_index(dim, 1, 1, 8, 32, 0, false, false, 4);
        hnswlib::RaBitQHierarchicalNSW turbo_index(dim, 1, 1, 8, 32, 0, false, false, 4);
        seq_index.space().setIdentityRotation();
        turbo_index.space().setIdentityRotation();
        seq_index.space().setGlobalCenter(center.data());
        turbo_index.space().setGlobalCenter(center.data());
        turbo_index.space().set_code_layout(hnswlib::RaBitQCodeLayout::Turbo128);
        const std::vector<char> seq_encoded = seq_index.space().encodeVector(point.data());
        const std::vector<char> turbo_encoded = turbo_index.space().encodeVector(point.data());
        const void *seq_query = seq_index.space().prepare_query(query.data());
        const void *turbo_query = turbo_index.space().prepare_query(query.data());
        const float seq_distance = seq_index.space().query_distance(seq_query, seq_encoded.data());
        const float turbo_distance = turbo_index.space().query_distance(turbo_query, turbo_encoded.data());
        assert(std::fabs(seq_distance - turbo_distance) <= 1e-4f);
        seq_index.space().release_query(seq_query);
        turbo_index.space().release_query(turbo_query);
        if (dim == 128) {
            turbo_index.addPoint(point.data(), 0);
            const std::string path = "/tmp/rabitq_turbo128.index";
            turbo_index.saveIndex(path);
            hnswlib::RaBitQHierarchicalNSW loaded(dim, 1, 1, 8, 32, 0, false, false, 4);
            loaded.loadIndex(path, 1);
            assert(loaded.space().get_code_layout() == hnswlib::RaBitQCodeLayout::Turbo128);
            auto result = loaded.searchKnnPlainThenResidualRerank(query.data(), 1, 1);
            assert(!result.empty() && std::isfinite(result.top().first));
            std::remove(path.c_str());
            std::remove((path + ".rabitq").c_str());
        }
    }
}

int main() {
    test_asymmetric_four_bit_construction();
    test_symmetric_four_bit_construction();
    assert(!hnswlib::HierarchicalNSW<float>::baselineWouldAccept(1.0f, 1.0f, 10, 10));
    assert(hnswlib::HierarchicalNSW<float>::baselineWouldAccept(1.0f, 1.0f, 9, 10));
    assert(hnswlib::HierarchicalNSW<float>::baselineWouldAccept(0.9f, 1.0f, 10, 10));
    const uint8_t nibble_boundaries[] = {0, 3, 4, 7, 8, 11, 12, 15};
    for (uint8_t value : nibble_boundaries)
        assert(hnswlib::RaBitQSpace::primaryTopTwoBits(value) == value / 4U);
    test_residual_pack_roundtrip();
    test_multimeans_eager_lazy_equivalence();
    test_turbo128_layout();
    for (size_t bits : {size_t(4), size_t(8)}) {
        run_residual_bits_smoke(bits);
    }
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
                                      ((index.space().get_code_dim() +
                                        hnswlib::RaBitQSpace::kResidualBlockSize - 1U) /
                                      hnswlib::RaBitQSpace::kResidualBlockSize) * sizeof(float) +
                                      index.space().get_compact_code_bytes() +
                                      sizeof(uint8_t) +
                                      index.space().get_residual_code_bytes();
    assert(index.space().get_data_size() == expected_data_size);

    const std::vector<char> encoded = index.space().encodeVector(data.data());
    assert(index.space().get_compact_code_bytes() ==
           (index.space().get_code_dim() * hnswlib::RaBitQSpace::kTotalBits + 7U) / 8U);
    const auto *code = reinterpret_cast<const unsigned char *>(
        encoded.data() + sizeof(hnswlib::RaBitQSpace::EncodedHeader) +
            sizeof(hnswlib::RaBitQSpace::ShortCodeFactors) +
            sizeof(hnswlib::RaBitQSpace::ResidualCodeFactors) +
            ((index.space().get_code_dim() +
              hnswlib::RaBitQSpace::kResidualBlockSize - 1U) /
             hnswlib::RaBitQSpace::kResidualBlockSize) * sizeof(float));
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
    const float lightweight_short_lower_bound =
        index.space().compute_short_lower_bound(query_context, encoded.data());
    const hnswlib::DistanceInterval long_interval =
        index.space().compute_long_distance_interval(query_context, encoded.data());
    const hnswlib::DistanceInterval residual_interval =
        index.space().compute_residual_distance_interval(query_context, encoded.data(), long_interval.estimate);
    assert(std::isfinite(short_interval.estimate));
    assert(std::isfinite(short_interval.lower_bound));
    assert(std::isfinite(short_interval.upper_bound));
    assert(std::isfinite(lightweight_short_lower_bound));
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

    for (const void *point : points) {
        const float full = index.space().query_distance(query_context, point);
        const float lower = index.space().compute_short_lower_bound(query_context, point);
        const float lower_two = index.space().compute_two_bit_lower_bound(query_context, point);
        const float lower_two_scalar =
            index.space().compute_two_bit_lower_bound_scalar_for_test(query_context, point);
        assert(std::isfinite(full));
        assert(lower <= full + 1e-5f * std::max(1.0f, std::fabs(full)));
        assert(lower_two <= full + 1e-5f * std::max(1.0f, std::fabs(full)));
        assert(std::fabs(lower_two - lower_two_scalar) <
               1e-5f * std::max(1.0f, std::fabs(lower_two_scalar)));
        for (const float epsilon0 : {1.9f, 2.2f, 2.5f}) {
            const auto paper = index.space().compute_paper_prune_estimate(
                query_context, point, epsilon0);
            std::vector<uint8_t> msb(index.space().paper_msb_code_bytes());
            const auto sidecar_factors = index.space().extract_paper_prune_sidecar(
                point, msb.data());
            const auto sidecar_paper = index.space().compute_paper_prune_estimate_sidecar(
                query_context, msb.data(), sidecar_factors, epsilon0);
            assert(paper.valid);
            assert(sidecar_paper.valid);
            assert(std::fabs(sidecar_paper.short_ip - paper.short_ip) < 1e-5f);
            assert(std::fabs(sidecar_paper.lower_bound - paper.lower_bound) <
                   1e-5f * std::max(1.0f, std::fabs(paper.lower_bound)));
            assert(paper.alpha > 0.0f && paper.alpha <= 1.0f);
            const float reference_error =
                std::sqrt((1.0f - paper.alpha * paper.alpha) /
                          (paper.alpha * paper.alpha)) *
                epsilon0 / std::sqrt(static_cast<float>(index.space().get_code_dim() - 1));
            assert(std::fabs(paper.error_bound - reference_error) < 1e-6f);
            const float staged = index.space().query_distance_with_paper_msb(
                query_context, point, paper.short_ip);
            assert(std::fabs(staged - full) <
                   1e-5f * std::max(1.0f, std::fabs(full)));
        }
    }
    const float one_distance = index.space().query_distance(query_context, points[0]);
    index.space().release_query(query_context);
    assert(std::fabs(single_distance - one_distance) < 1e-5f);

    for (size_t i = 0; i < 4; ++i) {
        index.addPoint(data.data() + i * dim, i);
    }

    auto result = index.searchKnn(data.data(), 1);
    assert(!result.empty());
    assert(std::isfinite(result.top().first));
    auto plain_rerank_result = index.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!plain_rerank_result.empty());
    assert(std::isfinite(plain_rerank_result.top().first));
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
    auto loaded_plain_rerank =
        loaded.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!loaded_plain_rerank.empty());
    assert(std::isfinite(loaded_plain_rerank.top().first));
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

    hnswlib::RaBitQHierarchicalNSW bfs_reordered_index(dim, 4, 1, 8, 32, 0);
    bfs_reordered_index.importBfsReorderedFrom(payload_build_index);
    assert(bfs_reordered_index.labelGraphFingerprint() ==
           payload_build_index.labelGraphFingerprint());
    assert(bfs_reordered_index.payloadFingerprintByLabel() ==
           payload_build_index.payloadFingerprintByLabel());
    assert(bfs_reordered_index.index().getExternalLabel(0) ==
           payload_build_index.index().getExternalLabel(
               payload_build_index.index().enterpoint_node_));
    const char *tmp_bfs_index = "/tmp/rabitq_hnsw_smoke_bfs.index";
    const char *tmp_bfs_state = "/tmp/rabitq_hnsw_smoke_bfs.index.rabitq";
    std::remove(tmp_bfs_index);
    std::remove(tmp_bfs_state);
    bfs_reordered_index.saveIndex(tmp_bfs_index);
    hnswlib::RaBitQHierarchicalNSW loaded_bfs(dim, 4, 1, 8, 32, 0);
    loaded_bfs.loadIndex(tmp_bfs_index, 4);
    assert(loaded_bfs.labelGraphFingerprint() == payload_build_index.labelGraphFingerprint());
    assert(loaded_bfs.payloadFingerprintByLabel() == payload_build_index.payloadFingerprintByLabel());
    auto bfs_result = loaded_bfs.searchKnn(data.data(), 1);
    assert(!bfs_result.empty());
    assert(bfs_result.top().second == payload_build_result.top().second);
    std::remove(tmp_bfs_index);
    std::remove(tmp_bfs_state);

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
    const char *tmp_floatbuild_route = "/tmp/rabitq_hnsw_smoke_floatbuild.index.route";
    std::remove(tmp_floatbuild_index);
    std::remove(tmp_floatbuild_state);
    std::remove(tmp_floatbuild_route);
    hnswlib::GraphTurboConfig turbo_config;
    turbo_config.mode = hnswlib::GraphTurboMode::BatchPrefetch;
    turbo_config.prefetch_distance = 2;
    payload_build_index.setGraphTurboConfig(turbo_config);
    hnswlib::RaBitQSearchMetrics batch_metrics;
    const auto batch_result = payload_build_index.searchKnnPlainThenResidualRerank(
        data.data(), 2, 2, nullptr, &batch_metrics);
    assert(batch_metrics.distance_computations > 0);
    turbo_config.short_shadow = true;
    payload_build_index.setEf(1);
    payload_build_index.setGraphTurboConfig(turbo_config);
    hnswlib::RaBitQSearchMetrics shadow_metrics;
    const auto shadow_result = payload_build_index.searchKnnPlainThenResidualRerank(
        data.data(), 1, 1, nullptr, &shadow_metrics);
    assert(!shadow_result.empty());
    assert(shadow_metrics.short_checked ==
           shadow_metrics.short_would_reject + shadow_metrics.short_ambiguous);
    assert(shadow_metrics.unsafe_reject == 0);
    assert(shadow_metrics.short_bound_violation == 0);
    assert(shadow_metrics.full_distance_count > 0);
    turbo_config.short_shadow = false;
    turbo_config.two_bit_shadow = true;
    payload_build_index.setGraphTurboConfig(turbo_config);
    hnswlib::RaBitQSearchMetrics two_bit_metrics;
    const auto two_bit_result = payload_build_index.searchKnnPlainThenResidualRerank(
        data.data(), 1, 1, nullptr, &two_bit_metrics);
    assert(!two_bit_result.empty());
    assert(two_bit_result.top().second == shadow_result.top().second);
    assert(two_bit_metrics.two_bit_checked ==
           two_bit_metrics.two_bit_would_reject + two_bit_metrics.two_bit_ambiguous);
    assert(two_bit_metrics.two_bit_unsafe_reject == 0);
    assert(two_bit_metrics.two_bit_bound_violation == 0);
    turbo_config.two_bit_shadow = false;
    turbo_config.paper_shadow = true;
    turbo_config.paper_epsilon0 = 1.9f;
    payload_build_index.setGraphTurboConfig(turbo_config);
    hnswlib::RaBitQSearchMetrics paper_shadow_metrics;
    const auto paper_shadow_result = payload_build_index.searchKnnPlainThenResidualRerank(
        data.data(), 1, 1, nullptr, &paper_shadow_metrics);
    assert(!paper_shadow_result.empty());
    assert(paper_shadow_result.top().second == shadow_result.top().second);
    assert(paper_shadow_metrics.paper_checked ==
           paper_shadow_metrics.paper_would_prune + paper_shadow_metrics.paper_not_pruned);
    turbo_config.paper_shadow = false;
    turbo_config.paper_active = true;
    payload_build_index.setGraphTurboConfig(turbo_config);
    hnswlib::RaBitQSearchMetrics paper_active_metrics;
    const auto paper_active_result = payload_build_index.searchKnnPlainThenResidualRerank(
        data.data(), 1, 1, nullptr, &paper_active_metrics);
    assert(!paper_active_result.empty());
    assert(paper_active_metrics.paper_checked ==
           paper_active_metrics.paper_would_prune + paper_active_metrics.paper_not_pruned);
    assert(paper_active_metrics.paper_remaining_kernel_calls +
           paper_active_metrics.paper_full_saved > 0);
    payload_build_index.buildRouteCodes(
        data.data(), 4, hnswlib::RouteCodeStrategy::EqualInterval, 6);
    assert(payload_build_index.routeCodeFingerprint() != 0);
    turbo_config.mode = hnswlib::GraphTurboMode::RoutePriority;
    turbo_config.short_shadow = false;
    turbo_config.two_bit_shadow = false;
    turbo_config.paper_active = false;
    turbo_config.route_bits = 6;
    turbo_config.top_p = 2;
    payload_build_index.setGraphTurboConfig(turbo_config);
    hnswlib::RaBitQSearchMetrics route_metrics;
    const auto route_result = payload_build_index.searchKnnPlainThenResidualRerank(
        data.data(), 2, 2, nullptr, &route_metrics);
    assert(route_metrics.route_scored > 0);
    assert(route_metrics.priority_full_distance_count +
           route_metrics.remaining_full_distance_count == route_metrics.route_scored);
    assert(batch_result.size() == route_result.size());
    bool frozen_rejected = false;
    try { payload_build_index.addPoint(data.data(), 99); }
    catch (const std::runtime_error &) { frozen_rejected = true; }
    assert(frozen_rejected);
    payload_build_index.saveIndex(tmp_floatbuild_index);
    hnswlib::RaBitQHierarchicalNSW loaded_floatbuild(dim, 4, 1, 8, 32, 0);
    loaded_floatbuild.loadIndex(tmp_floatbuild_index, 4);
    assert(loaded_floatbuild.routeCodeFingerprint() == payload_build_index.routeCodeFingerprint());
    loaded_floatbuild.setGraphTurboConfig(turbo_config);
    assert(loaded_floatbuild.index().data_size_ == loaded_floatbuild.space().get_data_size());
    auto loaded_floatbuild_result = loaded_floatbuild.searchKnn(data.data(), 1);
    assert(!loaded_floatbuild_result.empty());
    assert(loaded_floatbuild_result.top().second == payload_build_result.top().second);
    std::remove(tmp_floatbuild_index);
    std::remove(tmp_floatbuild_state);
    std::remove(tmp_floatbuild_route);

    const char *tmp_external_index = "/tmp/rabitq_hnsw_smoke_external.index";
    const char *tmp_external_state = "/tmp/rabitq_hnsw_smoke_external.index.rabitq";
    const char *tmp_external_residual = "/tmp/rabitq_hnsw_smoke_external.index.residual";
    const char *tmp_external_payload = "/tmp/rabitq_hnsw_smoke_external.payload";
    std::remove(tmp_external_index);
    std::remove(tmp_external_state);
    std::remove(tmp_external_residual);
    std::remove(tmp_external_payload);
    hnswlib::RaBitQHierarchicalNSW external_index(dim, 4, 1, 8, 32, 0, false, true);
    external_index.space().setIdentityRotation();
    {
        std::ofstream payload_output(tmp_external_payload, std::ios::binary);
        assert(payload_output.is_open());
        std::vector<char> full_encoded(external_index.space().get_full_data_size(), 0);
        for (size_t i = 0; i < 4; ++i) {
            external_index.space().encodeVectorFull(data.data() + i * dim, full_encoded.data());
            payload_output.write(full_encoded.data(), static_cast<std::streamsize>(full_encoded.size()));
        }
        assert(payload_output.good());
    }
    external_index.importGraphFromFloatIndexWithFullPayloadFileAndExternalResiduals(
        float_graph,
        tmp_external_payload,
        tmp_external_residual,
        external_index.space().get_full_data_size());
    auto external_result =
        external_index.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!external_result.empty());
    assert(std::isfinite(external_result.top().first));
    external_index.saveIndex(tmp_external_index);
    hnswlib::RaBitQHierarchicalNSW external_loaded(dim, 4, 1, 8, 32, 0, false, true);
    external_loaded.loadIndex(tmp_external_index, 4);
    auto external_loaded_result =
        external_loaded.searchKnnPlainThenResidualRerank(data.data(), 1, 4);
    assert(!external_loaded_result.empty());
    assert(std::isfinite(external_loaded_result.top().first));
    std::remove(tmp_external_index);
    std::remove(tmp_external_state);
    std::remove(tmp_external_residual);
    std::remove(tmp_external_payload);

    std::cout << "RaBitQ HNSW smoke test passed\n";
    return 0;
}
