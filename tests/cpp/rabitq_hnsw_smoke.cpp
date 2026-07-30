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
    std::remove(tmp_external_residual.c_str());
    std::remove(tmp_external_payload.c_str());
}

int main() {
    test_residual_pack_roundtrip();
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

    for (const void *point : points) {
        assert(std::isfinite(index.space().query_distance(query_context, point)));
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
