#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"

using namespace std;
using namespace hnswlib;

#ifndef RABITQ_EF_CONSTRUCTION
#define RABITQ_EF_CONSTRUCTION 200
#endif

namespace {

void print_run_config(
    const char *dataset_name,
    size_t vecsize,
    size_t qsize,
    size_t vecdim,
    int efConstruction,
    int M,
    int centroid_count,
    int rerank_candidates,
    int random_seed,
    const char *path_index,
    const char *path_data,
    const char *path_q,
    const char *path_gt) {
    (void) rerank_candidates;
    cout << "Run config:\n";
    cout << "  dataset=" << dataset_name << "\n";
    cout << "  base_count=" << vecsize << "\n";
    cout << "  query_count=" << qsize << "\n";
    cout << "  dimension=" << vecdim << "\n";
    cout << "  M=" << M << " efConstruction=" << efConstruction << "\n";
    cout << "  quantizer=4-bit ExRaBitQ centroid_count=" << centroid_count
         << " random_seed=" << random_seed << "\n";
    cout << "  build_distance=float32_l2"
         << " stored_data=4bit_rabitq query_distance=float32_query_to_4bit_code\n";
    cout << "  base_path=" << path_data << "\n";
    cout << "  query_path=" << path_q << "\n";
    cout << "  gt_path=" << path_gt << "\n";
    cout << "  index_path=" << path_index << "\n";
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

string quantizer_state_path(const string &index_path) {
    return index_path + ".rabitq";
}

size_t file_size_bytes(const string &path) {
    ifstream input(path, ios::binary | ios::ate);
    if (!input.is_open()) {
        return 0;
    }
    return static_cast<size_t>(input.tellg());
}

double kips_from_count_us(size_t count, double us) {
    return us > 0.0 ? static_cast<double>(count) / (1000.0 * us * 1e-6) : 0.0;
}

double mbps_from_bytes_us(size_t bytes, double us) {
    return us > 0.0 ? static_cast<double>(bytes) / 1000000.0 / (us * 1e-6) : 0.0;
}

void print_index_file_size(
    const string &index_path,
    size_t vecsize,
    size_t encoded_bytes_per_vector,
    const string &raw_rerank_path = "",
    bool raw_rerank_enabled = false) {
    const string state_path = quantizer_state_path(index_path);
    const size_t index_bytes = file_size_bytes(index_path);
    const size_t auxiliary_bytes = file_size_bytes(state_path);
    const size_t total_bytes = index_bytes + auxiliary_bytes;
    const size_t raw_rerank_bytes =
        raw_rerank_enabled ? file_size_bytes(raw_rerank_path) : 0;
    const size_t effective_query_storage_bytes = total_bytes + raw_rerank_bytes;
    const size_t encoded_payload_estimate_bytes = vecsize * encoded_bytes_per_vector;
    const size_t graph_and_metadata_bytes =
        total_bytes > encoded_payload_estimate_bytes ? total_bytes - encoded_payload_estimate_bytes : 0;
    const double mb = 1000000.0;
    cout << "Index storage size: " << total_bytes / mb << " MB"
         << " (index=" << index_bytes / mb << " MB"
         << ", auxiliary=" << auxiliary_bytes / mb << " MB"
         << ", total_bytes=" << total_bytes << ")\n";
    cout << "storage_breakdown"
         << " index_file_MB=" << index_bytes / mb
         << " auxiliary_MB=" << auxiliary_bytes / mb
         << " encoded_payload_estimate_MB=" << encoded_payload_estimate_bytes / mb
         << " graph_and_metadata_estimate_MB=" << graph_and_metadata_bytes / mb
         << " extra_rerank_data_MB=" << raw_rerank_bytes / mb
         << " raw_rerank_enabled=" << (raw_rerank_enabled ? 1 : 0)
         << " effective_query_storage_MB=" << effective_query_storage_bytes / mb
         << " effective_query_storage_bytes=" << effective_query_storage_bytes
         << "\n";
}

void read_bvec_as_float(ifstream &input, float *dst, size_t vecdim, vector<unsigned char> &scratch) {
    int in = 0;
    input.read((char *)&in, 4);
    if (!input.good() || in != static_cast<int>(vecdim)) {
        throw runtime_error("file error");
    }
    input.read((char *)scratch.data(), in);
    if (!input.good()) {
        throw runtime_error("file error");
    }
    for (size_t j = 0; j < vecdim; ++j) {
        dst[j] = static_cast<float>(scratch[j]);
    }
}

void read_bvec_as_float_and_u8(
    ifstream &input,
    float *float_dst,
    uint8_t *u8_dst,
    size_t vecdim,
    vector<unsigned char> &scratch) {
    read_bvec_as_float(input, float_dst, vecdim, scratch);
    std::copy(scratch.begin(), scratch.end(), u8_dst);
}

void read_bvec_as_u8(ifstream &input, uint8_t *dst, size_t vecdim) {
    int in = 0;
    input.read((char *)&in, 4);
    if (!input.good() || in != static_cast<int>(vecdim)) {
        throw runtime_error("file error");
    }
    input.read(reinterpret_cast<char *>(dst), in);
    if (!input.good()) {
        throw runtime_error("file error");
    }
}

vector<uint8_t> load_bvecs_raw(const string &path, size_t vec_count, size_t vecdim) {
    ifstream input(path, ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open raw bvec file: " + path);
    }

    vector<uint8_t> raw(vec_count * vecdim, 0);
    for (size_t i = 0; i < vec_count; ++i) {
        read_bvec_as_u8(input, raw.data() + i * vecdim, vecdim);
    }
    return raw;
}

float raw_l2_u8(const uint8_t *query, const uint8_t *base, size_t dim) {
    uint32_t total = 0;
    for (size_t i = 0; i < dim; ++i) {
        const int diff = static_cast<int>(query[i]) - static_cast<int>(base[i]);
        total += static_cast<uint32_t>(diff * diff);
    }
    return static_cast<float>(total);
}

float raw_l2_float(const float *query, const float *base, size_t dim) {
    float total = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float diff = query[i] - base[i];
        total += diff * diff;
    }
    return total;
}

void normalize_l2(float *data, size_t dim) {
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        norm += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    norm = std::sqrt(norm);
    if (norm == 0.0) {
        return;
    }
    for (size_t i = 0; i < dim; ++i) {
        data[i] = static_cast<float>(static_cast<double>(data[i]) / norm);
    }
}

size_t fvec_count_from_file_size(const string &path, size_t vecdim) {
    ifstream input(path, ios::binary | ios::ate);
    if (!input.is_open()) {
        throw runtime_error("cannot open fvec file: " + path);
    }
    const size_t bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int) + vecdim * sizeof(float);
    if (record_bytes == 0 || bytes % record_bytes != 0) {
        throw runtime_error("fvec file size is not divisible by record size: " + path);
    }
    return bytes / record_bytes;
}

void read_fvec_as_float(ifstream &input, float *dst, size_t vecdim) {
    int in = 0;
    input.read(reinterpret_cast<char *>(&in), 4);
    if (!input.good() || in != static_cast<int>(vecdim)) {
        throw runtime_error("file error");
    }
    input.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(vecdim * sizeof(float)));
    if (!input.good()) {
        throw runtime_error("file error");
    }
    normalize_l2(dst, vecdim);
}

vector<float> load_fvecs_raw(const string &path, size_t vec_count, size_t vecdim) {
    ifstream input(path, ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open raw fvec file: " + path);
    }

    vector<float> raw(vec_count * vecdim, 0.0f);
    for (size_t i = 0; i < vec_count; ++i) {
        read_fvec_as_float(input, raw.data() + i * vecdim, vecdim);
    }
    return raw;
}

class BVecRandomReader {
 private:
    ifstream input_;
    size_t vecdim_{0};
    size_t record_bytes_{0};
    vector<unsigned char> scratch_;

 public:
    BVecRandomReader(const string &path, size_t vecdim)
        : input_(path, ios::binary), vecdim_(vecdim), record_bytes_(4 + vecdim), scratch_(vecdim) {
        if (!input_.is_open()) {
            throw runtime_error("cannot open base file for rerank: " + path);
        }
    }

    void readVector(size_t label, float *dst) {
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(label * record_bytes_), ios::beg);
        if (!input_.good()) {
            throw runtime_error("failed to seek base vector");
        }
        read_bvec_as_float(input_, dst, vecdim_, scratch_);
    }
};

class FVecRandomReader {
 private:
    ifstream input_;
    size_t vecdim_{0};
    size_t record_bytes_{0};

 public:
    FVecRandomReader(const string &path, size_t vecdim)
        : input_(path, ios::binary), vecdim_(vecdim), record_bytes_(4 + vecdim * sizeof(float)) {
        if (!input_.is_open()) {
            throw runtime_error("cannot open base file for rerank: " + path);
        }
    }

    void readVector(size_t label, float *dst) {
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(label * record_bytes_), ios::beg);
        if (!input_.good()) {
            throw runtime_error("failed to seek base vector");
        }
        read_fvec_as_float(input_, dst, vecdim_);
    }
};

vector<float> train_kmeans_centroids(
    ifstream &input,
    size_t vecdim,
    size_t vecsize,
    size_t centroid_count,
    size_t sample_count,
    int random_seed,
    vector<unsigned char> &scratch) {
    if (centroid_count == 0) {
        throw runtime_error("centroid_count must be positive");
    }

    input.clear();
    input.seekg(0, ios::beg);

    const size_t actual_sample_count = min(sample_count, vecsize);
    if (actual_sample_count < centroid_count) {
        throw runtime_error("not enough samples to train centroids");
    }

    vector<float> samples(actual_sample_count * vecdim, 0.0f);
    for (size_t i = 0; i < actual_sample_count; ++i) {
        read_bvec_as_float(input, samples.data() + i * vecdim, vecdim, scratch);
    }

    std::mt19937 rng(random_seed);
    vector<size_t> init_ids(actual_sample_count);
    iota(init_ids.begin(), init_ids.end(), 0);
    shuffle(init_ids.begin(), init_ids.end(), rng);

    vector<float> centroids(centroid_count * vecdim, 0.0f);
    for (size_t centroid_id = 0; centroid_id < centroid_count; ++centroid_id) {
        const float *src = samples.data() + init_ids[centroid_id] * vecdim;
        copy(src, src + vecdim, centroids.data() + centroid_id * vecdim);
    }

    vector<float> next_centroids(centroid_count * vecdim, 0.0f);
    vector<size_t> counts(centroid_count, 0);
    const size_t kmeans_iters = 8;

    for (size_t iter = 0; iter < kmeans_iters; ++iter) {
        fill(next_centroids.begin(), next_centroids.end(), 0.0f);
        fill(counts.begin(), counts.end(), 0);

        for (size_t sample_id = 0; sample_id < actual_sample_count; ++sample_id) {
            const float *sample = samples.data() + sample_id * vecdim;
            size_t best_centroid = 0;
            float best_dist = numeric_limits<float>::max();
            for (size_t centroid_id = 0; centroid_id < centroid_count; ++centroid_id) {
                const float *centroid = centroids.data() + centroid_id * vecdim;
                float dist = 0.0f;
                for (size_t d = 0; d < vecdim; ++d) {
                    const float diff = sample[d] - centroid[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_centroid = centroid_id;
                }
            }

            ++counts[best_centroid];
            float *dst = next_centroids.data() + best_centroid * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] += sample[d];
            }
        }

        for (size_t centroid_id = 0; centroid_id < centroid_count; ++centroid_id) {
            float *dst = next_centroids.data() + centroid_id * vecdim;
            if (counts[centroid_id] == 0) {
                const size_t fallback_id = init_ids[centroid_id % actual_sample_count];
                const float *fallback = samples.data() + fallback_id * vecdim;
                copy(fallback, fallback + vecdim, dst);
                continue;
            }

            const float inv_count = 1.0f / static_cast<float>(counts[centroid_id]);
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] *= inv_count;
            }
        }

        centroids.swap(next_centroids);
    }

    input.clear();
    input.seekg(0, ios::beg);
    return centroids;
}

vector<float> train_global_center(
    ifstream &input,
    size_t vecdim,
    size_t vecsize,
    size_t sample_count) {
    input.clear();
    input.seekg(0, ios::beg);

    const size_t actual_sample_count = min(sample_count, vecsize);
    if (actual_sample_count == 0) {
        throw runtime_error("not enough samples to train global center");
    }

    vector<float> center(vecdim, 0.0f);
    vector<float> sample(vecdim, 0.0f);
    for (size_t i = 0; i < actual_sample_count; ++i) {
        read_fvec_as_float(input, sample.data(), vecdim);
        for (size_t d = 0; d < vecdim; ++d) {
            center[d] += sample[d];
        }
    }

    const float inv_count = 1.0f / static_cast<float>(actual_sample_count);
    for (float &value : center) {
        value *= inv_count;
    }

    input.clear();
    input.seekg(0, ios::beg);
    return center;
}

}  // namespace

class StopW {
    std::chrono::steady_clock::time_point time_begin;

 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count();
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

static size_t getCurrentRSS() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
#elif defined(__APPLE__) && defined(__MACH__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;
    return (size_t)info.resident_size;
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t)0L;
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
#else
    return (size_t)0L;
#endif
}

static void get_gt(
    unsigned int *massQA,
    size_t qsize,
    size_t gt_width,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<float, labeltype>>>(qsize)).swap(answers);
    cout << qsize << "\n";
    for (size_t i = 0; i < qsize; i++) {
        for (size_t j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_width * i + j]);
        }
    }
}

struct SearchReport {
    float recall{0.0f};
    float hnsw_search_us_per_query{0.0f};
    float redundant_rerank_us_per_query{0.0f};
    float total_us_per_query{0.0f};
    long lower_bound_checked{0};
    long lower_bound_pruned{0};
    long survivor_long_computed{0};
    float prune_rate{0.0f};
    float long_computations_per_query{0.0f};
};

static SearchReport test_approx(
    float *massQ,
    size_t qsize,
    RaBitQHierarchicalNSW &appr_alg,
    const string &base_path,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    size_t search_ef,
    size_t rerank_candidates) {
    size_t correct = 0;
    size_t total = 0;
    double hnsw_us = 0.0;
    double rerank_us = 0.0;
    const size_t candidate_count = std::max(k, std::min(rerank_candidates, std::max(k, search_ef)));
    FVecRandomReader raw_reader(base_path, vecdim);
    vector<float> raw_candidate(vecdim, 0.0f);
    appr_alg.index().metric_lower_bound_checked = 0;
    appr_alg.index().metric_lower_bound_pruned = 0;
    appr_alg.index().metric_survivor_long_computed = 0;

    for (size_t i = 0; i < qsize; i++) {
        StopW hnsw_timer;
        vector<pair<float, labeltype>> results =
            appr_alg.searchKnnCloserFirst(massQ + vecdim * i, candidate_count);
        hnsw_us += hnsw_timer.getElapsedTimeMicro();

        if (candidate_count > k) {
            StopW rerank_timer;
            vector<pair<float, labeltype>> reranked;
            reranked.reserve(results.size());
            const float *query = massQ + vecdim * i;
            for (const auto &entry : results) {
                raw_reader.readVector(static_cast<size_t>(entry.second), raw_candidate.data());
                reranked.emplace_back(raw_l2_float(query, raw_candidate.data(), vecdim), entry.second);
            }
            std::sort(reranked.begin(), reranked.end());
            if (reranked.size() > k) {
                reranked.resize(k);
            }
            results.swap(reranked);
            rerank_us += rerank_timer.getElapsedTimeMicro();
        } else if (results.size() > k) {
            results.resize(k);
        }

        std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (!gt.empty()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for (const auto &entry : results) {
            if (g.find(entry.second) != g.end()) {
                correct++;
            }
        }
    }

    SearchReport report;
    report.recall = total == 0 ? 0.0f : 1.0f * correct / total;
    report.hnsw_search_us_per_query = static_cast<float>(hnsw_us / static_cast<double>(qsize));
    report.redundant_rerank_us_per_query = static_cast<float>(rerank_us / static_cast<double>(qsize));
    report.total_us_per_query = report.hnsw_search_us_per_query + report.redundant_rerank_us_per_query;
    report.lower_bound_checked = appr_alg.index().metric_lower_bound_checked.load();
    report.lower_bound_pruned = appr_alg.index().metric_lower_bound_pruned.load();
    report.survivor_long_computed = appr_alg.index().metric_survivor_long_computed.load();
    report.long_computations_per_query =
        static_cast<float>(static_cast<double>(report.survivor_long_computed) /
                           static_cast<double>(qsize));
    report.prune_rate = report.lower_bound_checked == 0
                            ? 0.0f
                            : static_cast<float>(
                                  static_cast<double>(report.lower_bound_pruned) /
                                  static_cast<double>(report.lower_bound_checked));
    return report;
}

static void test_vs_recall(
    float *massQ,
    size_t qsize,
    RaBitQHierarchicalNSW &appr_alg,
    const string &base_path,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    size_t rerank_candidates) {
    vector<size_t> efs;
    for (size_t i = 1; i <= 30; i++) {
        if (i >= k) {
            efs.push_back(i);
        }
    }
    for (size_t i = 40; i <= 100; i += 10) {
        if (i >= k) {
            efs.push_back(i);
        }
    }
    for (size_t i = 140; i <= 460; i += 40) {
        if (i >= k) {
            efs.push_back(i);
        }
    }

    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        SearchReport report = test_approx(
            massQ,
            qsize,
            appr_alg,
            base_path,
            vecdim,
            answers,
            k,
            ef,
            rerank_candidates);

        cout << ef << "\t" << report.recall
             << "\t" << report.total_us_per_query << " us"
             << "\t" << "hnsw_search_us_per_query=" << report.hnsw_search_us_per_query
             << "\t" << "raw_rerank_us_per_query=" << report.redundant_rerank_us_per_query
             << "\t" << "rerank_candidates=" << std::max(k, std::min(rerank_candidates, std::max(k, ef)))
             << "\t" << "total_us_per_query=" << report.total_us_per_query << "\n";
        if (report.recall > 1.0f) {
            cout << report.recall << "\t" << report.total_us_per_query << " us\n";
            break;
        }
    }
}

void sift_test1B() {
    const char *dataset_name = "deep1B";
    const int efConstruction = RABITQ_EF_CONSTRUCTION;
    const int M = 16;
    const int centroid_count = 64;
    const int rerank_candidates = 100;
    const size_t centroid_train_samples = 200000;
    const int random_seed = 100;

    const size_t vecdim = 96;
    const size_t gt_width = 100;

    char path_index[1024];
    const char *path_q = "/home/kai3/coco/data/deep1B/deep1B_query.fvecs";
    const char *path_data = "/home/kai3/coco/data/deep1B/deep1B_base.fvecs";
    const char *path_gt = "/home/kai3/coco/data/deep1B/deep1B_groundtruth.ivecs";
    const size_t vecsize = fvec_count_from_file_size(path_data, vecdim);
    const size_t qsize = fvec_count_from_file_size(path_q, vecdim);
    snprintf(
        path_index,
        sizeof(path_index),
        "deep1B_rabitq_floatbuild_ef_%d_M_%d_C_%d.bin",
        efConstruction,
        M,
        centroid_count);

    print_run_config(
        dataset_name,
        vecsize,
        qsize,
        vecdim,
        efConstruction,
        M,
        centroid_count,
        rerank_candidates,
        random_seed,
        path_index,
        path_data,
        path_q,
        path_gt);

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    if (!inputGT.is_open()) {
        throw runtime_error("cannot open gt file");
    }
    unsigned int *massQA = new unsigned int[qsize * gt_width];
    for (size_t i = 0; i < qsize; i++) {
        int t = 0;
        inputGT.read((char *)&t, 4);
        if (!inputGT.good()) {
            throw runtime_error("gt file error");
        }
        if (t != static_cast<int>(gt_width)) {
            throw runtime_error("gt file error");
        }
        inputGT.read((char *)(massQA + gt_width * i), t * 4);
        if (!inputGT.good()) {
            throw runtime_error("gt file error");
        }
    }
    inputGT.close();

    cout << "Loading queries:\n";
    float *massQ = new float[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);
    if (!inputQ.is_open()) {
        throw runtime_error("cannot open query file");
    }
    for (size_t i = 0; i < qsize; i++) {
        read_fvec_as_float(inputQ, massQ + i * vecdim, vecdim);
    }
    inputQ.close();

    ifstream input(path_data, ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open base file");
    }

    RaBitQHierarchicalNSW *appr_alg = new RaBitQHierarchicalNSW(
        vecdim, vecsize, centroid_count, M, efConstruction, random_seed);
    cout << "  encoded_bytes_per_vector=" << appr_alg->space().get_data_size()
         << " (4-bit code)\n";

    bool need_build = true;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        if (!exists_test(quantizer_state_path(path_index))) {
            cout << "Missing RaBitQ quantizer state sidecar; rebuilding the index\n";
        } else {
            try {
                appr_alg->loadIndex(path_index, vecsize);
                print_index_file_size(
                    path_index,
                    vecsize,
                    appr_alg->space().get_data_size(),
                    path_data,
                    rerank_candidates > 1);
                need_build = false;
            } catch (const std::exception &error) {
                cout << "Existing index is incompatible: " << error.what() << "\n";
                cout << "Rebuilding the index with the current quantizer format\n";
                delete appr_alg;
                appr_alg = new RaBitQHierarchicalNSW(
                    vecdim, vecsize, centroid_count, M, efConstruction, random_seed);
                cout << "  encoded_bytes_per_vector=" << appr_alg->space().get_data_size()
                     << " (4-bit code)\n";
                input.clear();
                input.seekg(0, ios::beg);
            }
        }
    }

    if (need_build) {
        StopW total_build_timer;
        cout << "Building index:\n";
        cout << "Training one global ExRaBitQ center from "
             << min(centroid_train_samples, vecsize) << " base vectors\n";
        StopW train_center_timer;
        vector<float> global_center = train_global_center(
            input,
            vecdim,
            vecsize,
            centroid_train_samples);
        appr_alg->space().setGlobalCenter(global_center.data());
        const double train_center_us = train_center_timer.getElapsedTimeMicro();
        cout << "build_stage=train_center"
             << " us=" << train_center_us << "\n";

        int j1 = 0;
        StopW stopw;
        StopW float_graph_timer;
        const size_t report_every = 100000;
        const size_t payload_record_size = appr_alg->space().get_data_size();
        const size_t payload_total_bytes = vecsize * payload_record_size;
        vector<char> payloads(payload_total_bytes, 0);
        double payload_encode_cpu_us = 0.0;
        cout << "payload_mode=memory_array"
             << " payload_record_bytes=" << payload_record_size
             << " payload_bytes=" << payload_total_bytes << "\n";

        cout << "Building HNSW graph with float32 L2 distances, then encoding payloads to 4-bit RaBitQ\n";
        L2Space float_space(vecdim);
        HierarchicalNSW<float> float_index(
            &float_space,
            vecsize,
            M,
            efConstruction,
            random_seed);

        vector<float> first(vecdim);
        read_fvec_as_float(input, first.data(), vecdim);
        {
            StopW payload_timer;
            appr_alg->space().encodeVector(first.data(), payloads.data());
            payload_encode_cpu_us += payload_timer.getElapsedTimeMicro();
        }
        float_index.addPoint(first.data(), (size_t)0);

        StopW payload_encode_wall_timer;
#pragma omp parallel for
        for (int i = 1; i < static_cast<int>(vecsize); i++) {
            vector<float> local_mass(vecdim);
            int label = 0;
#pragma omp critical
            {
                read_fvec_as_float(input, local_mass.data(), vecdim);
                j1++;
                label = j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                         << " Mem: " << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            StopW payload_timer;
            appr_alg->space().encodeVector(
                local_mass.data(),
                payloads.data() + static_cast<size_t>(label) * payload_record_size);
            const double local_payload_encode_us = payload_timer.getElapsedTimeMicro();
#pragma omp atomic
            payload_encode_cpu_us += local_payload_encode_us;
            float_index.addPoint(local_mass.data(), (size_t)label);
        }
        const double payload_encode_wall_us = payload_encode_wall_timer.getElapsedTimeMicro();

        input.close();
        const double float_graph_build_us = float_graph_timer.getElapsedTimeMicro();
        cout << "Float graph build time:" << 1e-6 * float_graph_build_us
             << " seconds; importing graph and writing 4-bit payloads\n";
        cout << "build_stage=float_graph_build"
             << " us=" << float_graph_build_us
             << " kips=" << kips_from_count_us(vecsize, float_graph_build_us)
             << "\n";
        cout << "build_stage=payload_encode"
             << " cpu_us=" << payload_encode_cpu_us
             << " wall_us=" << payload_encode_wall_us
             << " count=" << vecsize
             << " record_bytes=" << payload_record_size
             << " total_bytes=" << payload_total_bytes
             << " cpu_kips=" << kips_from_count_us(vecsize, payload_encode_cpu_us)
             << " wall_kips=" << kips_from_count_us(vecsize, payload_encode_wall_us)
             << " cpu_MBps=" << mbps_from_bytes_us(payload_total_bytes, payload_encode_cpu_us)
             << " wall_MBps=" << mbps_from_bytes_us(payload_total_bytes, payload_encode_wall_us)
             << "\n";

        StopW convertw;
        appr_alg->importGraphFromFloatIndexWithPayloads(
            float_index,
            payloads,
            payload_record_size,
            true);
        const double graph_payload_import_us = convertw.getElapsedTimeMicro();
        const size_t graph_payload_import_count = float_index.cur_element_count;
        cout << "Float graph payload import time:" << 1e-6 * graph_payload_import_us
             << " seconds\n";
        cout << "build_stage=graph_payload_import"
             << " us=" << graph_payload_import_us
             << " count=" << graph_payload_import_count
             << " record_bytes=" << payload_record_size
             << " total_bytes=" << payload_total_bytes
             << " kips=" << kips_from_count_us(graph_payload_import_count, graph_payload_import_us)
             << " MBps=" << mbps_from_bytes_us(payload_total_bytes, graph_payload_import_us)
             << "\n";

        cout << "Build time:" << 1e-6 * total_build_timer.getElapsedTimeMicro() << "  seconds\n";
        StopW save_index_timer;
        appr_alg->saveIndex(path_index);
        const double save_index_us = save_index_timer.getElapsedTimeMicro();
        cout << "build_stage=save_index"
             << " us=" << save_index_us << "\n";
        cout << "build_total_us=" << total_build_timer.getElapsedTimeMicro() << "\n";
        print_index_file_size(
            path_index,
            vecsize,
            appr_alg->space().get_data_size(),
            path_data,
            rerank_candidates > 1);
    }

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    const size_t k = 1;
    cout << "Parsing gt:\n";
    get_gt(massQA, qsize, gt_width, answers, k);
    cout << "Loaded gt\n";
    test_vs_recall(
        massQ,
        qsize,
        *appr_alg,
        path_data,
        vecdim,
        answers,
        k,
        rerank_candidates);
    print_index_file_size(
        path_index,
        vecsize,
        appr_alg->space().get_data_size(),
        path_data,
        rerank_candidates > k);
}
