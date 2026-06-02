#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_rabitq.h"

using namespace std;
using namespace hnswlib;

namespace {

struct Timer {
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    double elapsedSeconds() const {
        return chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - begin).count();
    }
};

void read_bvec_as_float(ifstream &input, float *dst, size_t vecdim, vector<unsigned char> &scratch) {
    int dims = 0;
    input.read(reinterpret_cast<char *>(&dims), 4);
    if (!input.good() || dims != static_cast<int>(vecdim)) {
        throw runtime_error("bvec dimension mismatch or truncated file");
    }
    input.read(reinterpret_cast<char *>(scratch.data()), dims);
    if (!input.good()) {
        throw runtime_error("failed to read bvec payload");
    }
    for (size_t i = 0; i < vecdim; ++i) {
        dst[i] = static_cast<float>(scratch[i]);
    }
}

size_t read_bvec_dataset(const string &path, size_t max_count, size_t vecdim, vector<float> &data) {
    ifstream input(path, ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open file: " + path);
    }

    vector<unsigned char> scratch(vecdim);
    data.assign(max_count * vecdim, 0.0f);

    size_t count = 0;
    while (count < max_count) {
        const streampos before = input.tellg();
        int dims = 0;
        input.read(reinterpret_cast<char *>(&dims), 4);
        if (!input.good()) {
            break;
        }
        input.seekg(before);
        read_bvec_as_float(input, data.data() + count * vecdim, vecdim, scratch);
        ++count;
    }

    data.resize(count * vecdim);
    return count;
}

vector<float> train_kmeans_centroids(
    const vector<float> &base,
    size_t base_count,
    size_t vecdim,
    size_t centroid_count,
    size_t train_samples,
    uint32_t random_seed) {
    if (centroid_count == 0) {
        throw runtime_error("centroid_count must be positive");
    }
    const size_t sample_count = min(base_count, train_samples);
    if (sample_count < centroid_count) {
        throw runtime_error("train_samples must be >= centroid_count");
    }

    mt19937 rng(random_seed);
    vector<size_t> sample_ids(base_count);
    iota(sample_ids.begin(), sample_ids.end(), 0);
    shuffle(sample_ids.begin(), sample_ids.end(), rng);
    sample_ids.resize(sample_count);

    vector<float> centroids(centroid_count * vecdim, 0.0f);
    for (size_t c = 0; c < centroid_count; ++c) {
        const float *src = base.data() + sample_ids[c] * vecdim;
        copy(src, src + vecdim, centroids.data() + c * vecdim);
    }

    vector<float> next_centroids(centroid_count * vecdim, 0.0f);
    vector<size_t> counts(centroid_count, 0);
    const size_t kmeans_iters = 8;

    for (size_t iter = 0; iter < kmeans_iters; ++iter) {
        fill(next_centroids.begin(), next_centroids.end(), 0.0f);
        fill(counts.begin(), counts.end(), 0);

        for (size_t sid = 0; sid < sample_count; ++sid) {
            const float *sample = base.data() + sample_ids[sid] * vecdim;
            size_t best_centroid = 0;
            float best_dist = numeric_limits<float>::max();
            for (size_t c = 0; c < centroid_count; ++c) {
                const float *centroid = centroids.data() + c * vecdim;
                float dist = 0.0f;
                for (size_t d = 0; d < vecdim; ++d) {
                    const float diff = sample[d] - centroid[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_centroid = c;
                }
            }

            ++counts[best_centroid];
            float *dst = next_centroids.data() + best_centroid * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] += sample[d];
            }
        }

        for (size_t c = 0; c < centroid_count; ++c) {
            float *dst = next_centroids.data() + c * vecdim;
            if (counts[c] == 0) {
                const float *fallback = base.data() + sample_ids[c % sample_count] * vecdim;
                copy(fallback, fallback + vecdim, dst);
                continue;
            }
            const float inv_count = 1.0f / static_cast<float>(counts[c]);
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] *= inv_count;
            }
        }

        centroids.swap(next_centroids);
    }

    return centroids;
}

float l2sqr(const float *lhs, const float *rhs, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float diff = lhs[i] - rhs[i];
        dist += diff * diff;
    }
    return dist;
}

vector<labeltype> top_r_exact(const vector<float> &base, size_t base_count, const float *query, size_t dim, size_t top_r) {
    vector<pair<float, labeltype>> scored;
    scored.reserve(base_count);
    for (size_t i = 0; i < base_count; ++i) {
        scored.emplace_back(l2sqr(base.data() + i * dim, query, dim), static_cast<labeltype>(i));
    }
    partial_sort(scored.begin(), scored.begin() + top_r, scored.end());

    vector<labeltype> result;
    result.reserve(top_r);
    for (size_t i = 0; i < top_r; ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

vector<labeltype> top_r_rabitq(
    const vector<vector<char>> &encoded_base,
    const vector<char> &encoded_query,
    const RaBitQSpace &space,
    size_t top_r) {
    vector<pair<float, labeltype>> scored;
    scored.reserve(encoded_base.size());
    RaBitQSpace *mutable_space = const_cast<RaBitQSpace *>(&space);
    const auto dist_func = mutable_space->get_dist_func();
    void *dist_param = mutable_space->get_dist_func_param();
    for (size_t i = 0; i < encoded_base.size(); ++i) {
        scored.emplace_back(
            dist_func(encoded_query.data(), encoded_base[i].data(), dist_param),
            static_cast<labeltype>(i));
    }
    partial_sort(scored.begin(), scored.begin() + top_r, scored.end());

    vector<labeltype> result;
    result.reserve(top_r);
    for (size_t i = 0; i < top_r; ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

double overlap_recall(const vector<labeltype> &gt, const vector<labeltype> &approx, size_t k) {
    unordered_set<labeltype> gt_set;
    for (size_t i = 0; i < k; ++i) {
        gt_set.insert(gt[i]);
    }

    size_t hit = 0;
    for (labeltype id : approx) {
        if (gt_set.find(id) != gt_set.end()) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0]
             << " <base.bvecs> <query.bvecs> [base_count=100000] [query_count=100] [top_r=100] [centroid_count=64] [train_samples=20000]\n";
        return 1;
    }

    const string base_path = argv[1];
    const string query_path = argv[2];
    const size_t base_limit = argc > 3 ? stoull(argv[3]) : 100000;
    const size_t query_limit = argc > 4 ? stoull(argv[4]) : 100;
    const size_t top_r = argc > 5 ? stoull(argv[5]) : 100;
    const size_t centroid_count = argc > 6 ? stoull(argv[6]) : 64;
    const size_t train_samples = argc > 7 ? stoull(argv[7]) : 20000;
    const uint32_t random_seed = 100;
    const size_t vecdim = 128;

    cout << "Config:\n";
    cout << "  base_path=" << base_path << "\n";
    cout << "  query_path=" << query_path << "\n";
    cout << "  base_limit=" << base_limit << "\n";
    cout << "  query_limit=" << query_limit << "\n";
    cout << "  top_r=" << top_r << "\n";
    cout << "  centroid_count=" << centroid_count << "\n";
    cout << "  train_samples=" << train_samples << "\n";

    vector<float> base;
    vector<float> queries;

    Timer io_timer;
    const size_t base_count = read_bvec_dataset(base_path, base_limit, vecdim, base);
    const size_t query_count = read_bvec_dataset(query_path, query_limit, vecdim, queries);
    cout << "Loaded " << base_count << " base vectors and " << query_count
         << " queries in " << io_timer.elapsedSeconds() << " s\n";

    if (base_count == 0 || query_count == 0) {
        throw runtime_error("base_count and query_count must both be positive");
    }
    if (top_r == 0 || top_r > base_count) {
        throw runtime_error("top_r must be in [1, base_count]");
    }

    Timer train_timer;
    vector<float> centroids = train_kmeans_centroids(
        base, base_count, vecdim, centroid_count, train_samples, random_seed);
    RaBitQSpace rabitq_space(vecdim, centroid_count, random_seed);
    rabitq_space.setCentroids(centroids.data(), centroid_count);
    cout << "Trained centroids in " << train_timer.elapsedSeconds() << " s\n";

    Timer encode_timer;
    vector<vector<char>> encoded_base(base_count);
    for (size_t i = 0; i < base_count; ++i) {
        encoded_base[i] = rabitq_space.encodeVector(base.data() + i * vecdim);
    }
    cout << "Encoded base vectors in " << encode_timer.elapsedSeconds() << " s\n";

    double total_exact_secs = 0.0;
    double total_approx_secs = 0.0;
    double recall_at_1_sum = 0.0;
    double recall_at_10_sum = 0.0;
    double recall_at_r_sum = 0.0;

    const size_t recall_k10 = min<size_t>(10, top_r);

    for (size_t qid = 0; qid < query_count; ++qid) {
        const float *query = queries.data() + qid * vecdim;

        Timer exact_timer;
        vector<labeltype> gt_top_r = top_r_exact(base, base_count, query, vecdim, top_r);
        total_exact_secs += exact_timer.elapsedSeconds();

        const vector<char> encoded_query = rabitq_space.encodeVector(query);
        Timer approx_timer;
        vector<labeltype> approx_top_r = top_r_rabitq(encoded_base, encoded_query, rabitq_space, top_r);
        total_approx_secs += approx_timer.elapsedSeconds();

        recall_at_1_sum += overlap_recall(gt_top_r, approx_top_r, 1);
        recall_at_10_sum += overlap_recall(gt_top_r, approx_top_r, recall_k10);
        recall_at_r_sum += overlap_recall(gt_top_r, approx_top_r, top_r);
    }

    cout << "Results:\n";
    cout << "  exact_bruteforce_time_per_query_ms=" << (1000.0 * total_exact_secs / query_count) << "\n";
    cout << "  rabitq_bruteforce_time_per_query_ms=" << (1000.0 * total_approx_secs / query_count) << "\n";
    cout << "  recall@1_in_top" << top_r << "=" << (recall_at_1_sum / query_count) << "\n";
    cout << "  recall@" << recall_k10 << "_set_overlap=" << (recall_at_10_sum / query_count) << "\n";
    cout << "  recall@" << top_r << "_set_overlap=" << (recall_at_r_sum / query_count) << "\n";

    return 0;
}
