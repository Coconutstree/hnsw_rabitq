#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
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
                                      index.space().get_compact_code_bytes() +
                                      index.space().get_code_dim() / 8;
    assert(index.space().get_data_size() == expected_data_size);

    const std::vector<char> encoded = index.space().encodeVector(data.data());
    assert(index.space().get_compact_code_bytes() == index.space().get_code_dim() / 2);
    const auto *code = reinterpret_cast<const unsigned char *>(
        encoded.data() + sizeof(hnswlib::RaBitQSpace::EncodedHeader));
    for (size_t i = 0; i < index.space().get_compact_code_bytes(); ++i) {
        assert((code[i] & 0x0F) <= hnswlib::RaBitQSpace::kBaseMax);
        assert((code[i] >> 4) <= hnswlib::RaBitQSpace::kBaseMax);
    }

    const void *query_context = index.space().prepare_query(data.data());
    const float single_distance = index.space().query_distance(query_context, encoded.data());
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
            const float expected = index.space().query_distance(query_context, points[i]);
            assert(std::fabs(expected - batch_distance[i]) < 1e-4f);
        }
    }
    float one_distance[1] = {0.0f};
    index.space().batch_query_distance(query_context, &points[0], 1, one_distance);
    index.space().release_query(query_context);
    assert(std::fabs(single_distance - one_distance[0]) < 1e-5f);

    for (size_t i = 0; i < 4; ++i) {
        index.addPoint(data.data() + i * dim, i);
    }
    assert(index.space().rawStoreCount() == 4);

    auto result = index.searchKnn(data.data(), 1);
    assert(!result.empty());
    assert(result.top().second == 0);
    assert(std::fabs(result.top().first) < 1e-5f);

    const char *tmp_index = "/tmp/rabitq_hnsw_smoke.index";
    const char *tmp_state = "/tmp/rabitq_hnsw_smoke.index.rabitq";
    const char *tmp_raw = "/tmp/rabitq_hnsw_smoke.index.raw";
    std::remove(tmp_index);
    std::remove(tmp_state);
    std::remove(tmp_raw);
    index.saveIndex(tmp_index);
    hnswlib::RaBitQHierarchicalNSW loaded(dim, 4, 1, 8, 32, 0);
    loaded.loadIndex(tmp_index, 4);
    assert(loaded.space().rawStoreCount() == 4);
    auto loaded_result = loaded.searchKnn(data.data(), 1);
    assert(!loaded_result.empty());
    assert(loaded_result.top().second == result.top().second);
    assert(std::fabs(loaded_result.top().first - result.top().first) < 1e-5f);
    std::remove(tmp_index);
    std::remove(tmp_state);
    std::remove(tmp_raw);

    std::cout << "RaBitQ HNSW smoke test passed\n";
    return 0;
}
