#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../hnswlib/rabitq_hnsw.h"

using namespace hnswlib;

namespace {

class FvecFile {
 public:
    FvecFile(const std::string &path, size_t dim)
        : path_(path), dim_(dim), record_bytes_(sizeof(int) + dim * sizeof(float)), input_(path.c_str(), std::ios::binary) {
        if (!input_.is_open()) {
            throw std::runtime_error("cannot open fvec file: " + path);
        }
        input_.seekg(0, std::ios::end);
        const size_t bytes = static_cast<size_t>(input_.tellg());
        if (record_bytes_ == 0 || bytes % record_bytes_ != 0) {
            throw std::runtime_error("bad fvec file size: " + path);
        }
        count_ = bytes / record_bytes_;
    }

    size_t count() const {
        return count_;
    }

    void read(size_t id, std::vector<float> &dst) {
        if (id >= count_) {
            throw std::runtime_error("fvec id out of range: " + path_);
        }
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(id * record_bytes_), std::ios::beg);
        int stored_dim = 0;
        input_.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        if (!input_.good() || stored_dim != static_cast<int>(dim_)) {
            throw std::runtime_error("bad fvec record in: " + path_);
        }
        dst.assign(dim_, 0.0f);
        input_.read(reinterpret_cast<char *>(dst.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!input_.good()) {
            throw std::runtime_error("failed to read fvec record in: " + path_);
        }
    }

 private:
    std::string path_;
    size_t dim_;
    size_t record_bytes_;
    size_t count_{0};
    std::ifstream input_;
};

class IvecFile {
 public:
    IvecFile(const std::string &path, size_t width)
        : path_(path), width_(width), record_bytes_(sizeof(int) + width * sizeof(uint32_t)), input_(path.c_str(), std::ios::binary) {
        if (!input_.is_open()) {
            throw std::runtime_error("cannot open ivec file: " + path);
        }
    }

    std::vector<uint32_t> read(size_t id) {
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(id * record_bytes_), std::ios::beg);
        int stored_width = 0;
        input_.read(reinterpret_cast<char *>(&stored_width), sizeof(stored_width));
        if (!input_.good() || stored_width != static_cast<int>(width_)) {
            throw std::runtime_error("bad ivec record in: " + path_);
        }
        std::vector<uint32_t> ids(width_, 0);
        input_.read(reinterpret_cast<char *>(ids.data()), static_cast<std::streamsize>(width_ * sizeof(uint32_t)));
        if (!input_.good()) {
            throw std::runtime_error("failed to read ivec record in: " + path_);
        }
        return ids;
    }

 private:
    std::string path_;
    size_t width_;
    size_t record_bytes_;
    std::ifstream input_;
};

double recall_at(const std::vector<uint32_t> &truth, const std::vector<uint32_t> &predicted, size_t k) {
    std::unordered_set<uint32_t> truth_set;
    const size_t truth_count = std::min(k, truth.size());
    const size_t predicted_count = std::min(k, predicted.size());
    for (size_t i = 0; i < truth_count; ++i) {
        truth_set.insert(truth[i]);
    }

    size_t matches = 0;
    for (size_t i = 0; i < predicted_count; ++i) {
        if (truth_set.find(predicted[i]) != truth_set.end()) {
            ++matches;
        }
    }
    return truth_count > 0 ? static_cast<double>(matches) / static_cast<double>(truth_count) : 0.0;
}

struct RecallSums {
    double r1 = 0.0;
    double r10 = 0.0;
    double r100 = 0.0;
    size_t count = 0;

    void add(const std::vector<uint32_t> &truth, const std::vector<uint32_t> &predicted) {
        r1 += recall_at(truth, predicted, 1);
        r10 += recall_at(truth, predicted, 10);
        r100 += recall_at(truth, predicted, 100);
        ++count;
    }

    void print(const char *name) const {
        const double denom = static_cast<double>(count);
        std::cout << name << ":\n";
        std::cout << "  query_count=" << count << "\n";
        std::cout << "  recall@1=" << r1 / denom << "\n";
        std::cout << "  recall@10=" << r10 / denom << "\n";
        std::cout << "  recall@100=" << r100 / denom << "\n";
    }
};

std::vector<uint32_t> brute_force_quantized_topk(
    const std::vector<char> &encoded_query,
    RaBitQHierarchicalNSW &index,
    size_t k) {
    typedef std::pair<float, uint32_t> ScoredId;
    std::priority_queue<ScoredId> top;
    DISTFUNC<float> distance = index.space().get_dist_func();
    void *distance_param = index.space().get_dist_func_param();
    const size_t element_count = index.index().cur_element_count;

    for (tableint internal_id = 0; internal_id < static_cast<tableint>(element_count); ++internal_id) {
        const char *encoded_base = index.index().getDataByInternalId(internal_id);
        const float dist = distance(encoded_query.data(), encoded_base, distance_param);
        const uint32_t label = static_cast<uint32_t>(index.index().getExternalLabel(internal_id));
        if (top.size() < k) {
            top.emplace(dist, label);
        } else if (dist < top.top().first || (dist == top.top().first && label < top.top().second)) {
            top.pop();
            top.emplace(dist, label);
        }
    }

    std::vector<ScoredId> scored;
    scored.reserve(top.size());
    while (!top.empty()) {
        scored.push_back(top.top());
        top.pop();
    }
    std::sort(
        scored.begin(),
        scored.end(),
        [](const ScoredId &lhs, const ScoredId &rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

    std::vector<uint32_t> ids;
    ids.reserve(scored.size());
    for (const auto &item : scored) {
        ids.push_back(item.second);
    }
    return ids;
}

}  // namespace

int main() {
    const std::string dataset = "dbpedia_openai1536";
    const std::string query_path = "/home/kai3/coco/data/dbpedia_openai1536/dbpedia_openai1536_query.fvecs";
    const std::string gt_path = "/home/kai3/coco/data/dbpedia_openai1536/dbpedia_openai1536_groundtruth.ivecs";
    const std::string index_path = "build/dbpedia_openai1536_rabitq_current_ef_40_M_16_C_64.bin";
    const size_t dim = 1536;
    const size_t gt_width = 100;
    const size_t query_sample_count = 20;
    const size_t k = 100;

    FvecFile queries(query_path, dim);
    IvecFile groundtruth(gt_path, gt_width);
    if (queries.count() < query_sample_count) {
        throw std::runtime_error("not enough queries for requested sample count");
    }

    RaBitQHierarchicalNSW index(dim, 1, 64, 16, 40, 100);
    index.loadIndex(index_path);

    RecallSums recall;
    std::vector<float> query_vector;
    const auto start = std::chrono::steady_clock::now();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "RaBitQ full-database brute-force quantized recall check\n";
    std::cout << "  dataset=" << dataset << "\n";
    std::cout << "  quantizer=4-bit ExRaBitQ\n";
    std::cout << "  distance_under_test=encoded_query_to_encoded_data_full_scan\n";
    std::cout << "  reference=original_groundtruth_from_raw_l2\n";
    std::cout << "  indexed_base_count=" << index.index().cur_element_count << "\n";
    std::cout << "  query_count=" << queries.count() << "\n";
    std::cout << "  query_samples=" << query_sample_count << "\n";
    std::cout << "  topk=" << k << "\n";
    std::cout << "  encoded_bytes_per_vector=" << index.space().get_data_size() << "\n";
    std::cout << "  index_path=" << index_path << "\n";

    for (size_t query_id = 0; query_id < query_sample_count; ++query_id) {
        queries.read(query_id, query_vector);
        std::vector<char> encoded_query = index.space().encodeVector(query_vector.data());
        const std::vector<uint32_t> gt_ids = groundtruth.read(query_id);
        const std::vector<uint32_t> predicted = brute_force_quantized_topk(encoded_query, index, k);
        recall.add(gt_ids, predicted);
        std::cout << "  finished_query=" << query_id + 1
                  << " recall@100=" << recall_at(gt_ids, predicted, 100) << "\n";
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    recall.print("recall_vs_original_groundtruth_full_database");
    std::cout << "timing:\n";
    std::cout << "  elapsed_seconds=" << elapsed_seconds << "\n";
    std::cout << "  seconds_per_query=" << elapsed_seconds / static_cast<double>(query_sample_count) << "\n";
    return 0;
}
