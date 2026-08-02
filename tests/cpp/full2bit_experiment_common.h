#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../hnswlib/rabitq_hnsw.h"

namespace full2bit_exp {

using hnswlib::labeltype;

struct Config {
    std::string dataset = "sift10m";
    size_t dim = 128;
    size_t base_count = 10000000;
    size_t query_count = 1000;
    size_t gt_width = 1000;
    size_t centroid_count = 256;
    size_t m = 32;
    size_t ef_construction = 400;
    size_t random_seed = 100;
    std::string base_path = "/home/kai3/coco/data/sift10m/sift10m_base.fvecs";
    std::string query_path = "/home/kai3/coco/data/sift10m/sift10m_query.fvecs";
    std::string gt_path = "/home/kai3/coco/data/sift10m/sift10m_groundtruth.ivecs";
    std::string index_path =
        "build/trueK256_train10m_full2bit/sift10m_full2bit_trueK256_floatbuild_ef_400_M_32.bin";
};

inline std::string getenv_string(const char *name, const std::string &fallback) {
    const char *value = std::getenv(name);
    return value && value[0] ? std::string(value) : fallback;
}

inline size_t getenv_size_t(const char *name, size_t fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid integer env var: ") + name);
    }
    return static_cast<size_t>(parsed);
}

inline Config load_config() {
    Config cfg;
    cfg.dataset = getenv_string("RABITQ_DATASET", cfg.dataset);
    cfg.dim = getenv_size_t("RABITQ_DIM", cfg.dim);
    cfg.base_count = getenv_size_t("RABITQ_BASE_COUNT", cfg.base_count);
    cfg.query_count = getenv_size_t("RABITQ_QUERY_COUNT", cfg.query_count);
    cfg.gt_width = getenv_size_t("RABITQ_GT_WIDTH", cfg.gt_width);
    cfg.centroid_count = getenv_size_t("RABITQ_CENTROID_COUNT", cfg.centroid_count);
    cfg.m = getenv_size_t("RABITQ_M", cfg.m);
    cfg.ef_construction = getenv_size_t("RABITQ_EF_CONSTRUCTION", cfg.ef_construction);
    cfg.random_seed = getenv_size_t("RABITQ_RANDOM_SEED", cfg.random_seed);
    const std::string root = getenv_string("RABITQ_DATA_ROOT", "/home/kai3/coco/data/" + cfg.dataset);
    cfg.base_path = getenv_string("RABITQ_BASE_PATH", root + "/" + cfg.dataset + "_base.fvecs");
    cfg.query_path = getenv_string("RABITQ_QUERY_PATH", root + "/" + cfg.dataset + "_query.fvecs");
    cfg.gt_path = getenv_string("RABITQ_GT_PATH", root + "/" + cfg.dataset + "_groundtruth.ivecs");
    cfg.index_path = getenv_string("RABITQ_INDEX_PATH", cfg.index_path);
    return cfg;
}

inline hnswlib::RaBitQSpace::ResidualQuantizationConfig full2bit_config() {
    hnswlib::RaBitQSpace::ResidualQuantizationConfig cfg;
    cfg.bits = hnswlib::RaBitQSpace::ResidualQuantizationBits::B1;
    cfg.full2bit_baseline = true;
    return cfg;
}

inline std::unique_ptr<hnswlib::RaBitQHierarchicalNSW> load_full2bit_index(const Config &cfg) {
    std::unique_ptr<hnswlib::RaBitQHierarchicalNSW> index(
        new hnswlib::RaBitQHierarchicalNSW(
        cfg.dim,
        cfg.base_count,
        cfg.centroid_count,
        cfg.m,
        cfg.ef_construction,
        cfg.random_seed,
        false,
        false,
        full2bit_config()));
    index->loadIndex(cfg.index_path, cfg.base_count);
    if (!index->space().is_full2bit_baseline()) {
        throw std::runtime_error("loaded index is not full2bit baseline");
    }
    return index;
}

inline std::vector<float> read_fvecs(const std::string &path, size_t count, size_t dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open fvecs: " + path);
    }
    std::vector<float> vectors(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t stored_dim = 0;
        input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        if (!input || stored_dim != static_cast<int32_t>(dim)) {
            throw std::runtime_error("bad fvecs dimension while reading: " + path);
        }
        input.read(reinterpret_cast<char *>(vectors.data() + i * dim), dim * sizeof(float));
        if (!input) {
            throw std::runtime_error("truncated fvecs: " + path);
        }
    }
    return vectors;
}

inline std::vector<float> read_fvec_at(const std::string &path, size_t row, size_t dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open fvecs: " + path);
    }
    const std::streamoff record_bytes =
        static_cast<std::streamoff>(sizeof(int32_t) + dim * sizeof(float));
    input.seekg(static_cast<std::streamoff>(row) * record_bytes, std::ios::beg);
    int32_t stored_dim = 0;
    input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
    if (!input || stored_dim != static_cast<int32_t>(dim)) {
        throw std::runtime_error("bad fvecs row while reading: " + path);
    }
    std::vector<float> vector(dim);
    input.read(reinterpret_cast<char *>(vector.data()), dim * sizeof(float));
    if (!input) {
        throw std::runtime_error("truncated fvecs row: " + path);
    }
    return vector;
}

inline std::vector<std::vector<labeltype>> read_ivecs(
    const std::string &path,
    size_t count,
    size_t width) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open ivecs: " + path);
    }
    std::vector<std::vector<labeltype>> gt(count, std::vector<labeltype>(width));
    for (size_t i = 0; i < count; ++i) {
        int32_t stored_width = 0;
        input.read(reinterpret_cast<char *>(&stored_width), sizeof(stored_width));
        if (!input || stored_width < static_cast<int32_t>(width)) {
            throw std::runtime_error("bad ivecs width while reading: " + path);
        }
        for (size_t j = 0; j < static_cast<size_t>(stored_width); ++j) {
            int32_t id = 0;
            input.read(reinterpret_cast<char *>(&id), sizeof(id));
            if (j < width) {
                gt[i][j] = static_cast<labeltype>(id);
            }
        }
        if (!input) {
            throw std::runtime_error("truncated ivecs: " + path);
        }
    }
    return gt;
}

inline float l2_sqr(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline std::vector<std::pair<float, labeltype>> queue_to_sorted(
    std::priority_queue<std::pair<float, labeltype>> queue) {
    std::vector<std::pair<float, labeltype>> result;
    while (!queue.empty()) {
        result.push_back(queue.top());
        queue.pop();
    }
    std::sort(result.begin(), result.end());
    return result;
}

inline double recall_at(
    const std::vector<labeltype> &result,
    const std::vector<labeltype> &gt,
    size_t k) {
    const size_t limit = std::min(k, gt.size());
    std::unordered_set<labeltype> expected;
    expected.reserve(limit * 2);
    for (size_t i = 0; i < limit; ++i) {
        expected.insert(gt[i]);
    }
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, result.size()); ++i) {
        hits += expected.count(result[i]) != 0 ? 1U : 0U;
    }
    return limit == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(limit);
}

inline std::vector<labeltype> labels_from_pairs(
    const std::vector<std::pair<float, labeltype>> &pairs,
    size_t limit) {
    std::vector<labeltype> labels;
    labels.reserve(std::min(limit, pairs.size()));
    for (size_t i = 0; i < std::min(limit, pairs.size()); ++i) {
        labels.push_back(pairs[i].second);
    }
    return labels;
}

inline std::vector<size_t> parse_efs(const std::string &value, const std::vector<size_t> &fallback) {
    if (value.empty()) {
        return fallback;
    }
    std::vector<size_t> efs;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            efs.push_back(static_cast<size_t>(std::strtoull(item.c_str(), nullptr, 10)));
        }
    }
    return efs.empty() ? fallback : efs;
}

inline void print_config(const Config &cfg, const char *experiment) {
    std::cout << "Full2Bit experiment=" << experiment << "\n";
    std::cout << "dataset=" << cfg.dataset
              << " base_count=" << cfg.base_count
              << " query_count=" << cfg.query_count
              << " dim=" << cfg.dim
              << " K=" << cfg.centroid_count
              << " M=" << cfg.m
              << " efConstruction=" << cfg.ef_construction << "\n";
    std::cout << "index_path=" << cfg.index_path << "\n";
    std::cout << "base_path=" << cfg.base_path << "\n";
    std::cout << "query_path=" << cfg.query_path << "\n";
    std::cout << "gt_path=" << cfg.gt_path << "\n";
}

}  // namespace full2bit_exp
