#include <cassert>
#include <cmath>
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

    const std::vector<char> encoded = index.space().encodeVector(data.data());
    assert(index.space().get_compact_code_bytes() == index.space().get_code_dim() / 2);
    const auto *code = reinterpret_cast<const unsigned char *>(
        encoded.data() + sizeof(hnswlib::RaBitQSpace::EncodedHeader));
    for (size_t i = 0; i < index.space().get_compact_code_bytes(); ++i) {
        assert((code[i] & 0x0F) <= hnswlib::RaBitQSpace::kBaseMax);
        assert((code[i] >> 4) <= hnswlib::RaBitQSpace::kBaseMax);
    }

    const void *query_context = index.space().prepare_query(data.data());
    const void *points[1] = {encoded.data()};
    float batch_distance[1] = {0.0f};
    const float single_distance = index.space().query_distance(query_context, encoded.data());
    index.space().batch_query_distance(query_context, points, 1, batch_distance);
    index.space().release_query(query_context);
    assert(std::fabs(single_distance - batch_distance[0]) < 1e-5f);

    for (size_t i = 0; i < 4; ++i) {
        index.addPoint(data.data() + i * dim, i);
    }

    auto result = index.searchKnn(data.data(), 1);
    assert(!result.empty());
    assert(result.top().second == 0);

    std::cout << "RaBitQ HNSW smoke test passed\n";
    return 0;
}
