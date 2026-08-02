#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"

using namespace std;
using namespace hnswlib;

namespace {

enum class QueryMode {
    ResidualRerank,
    Full2Bit,
    Primary1Bit,
    Progressive1p1Bit,
    Full2BitBlockwise,
};

const char *query_mode_name(QueryMode mode) {
    if (mode == QueryMode::Full2Bit) {
        return "full2bit";
    }
    if (mode == QueryMode::Primary1Bit) {
        return "primary1bit";
    }
    if (mode == QueryMode::Progressive1p1Bit) {
        return "progressive1p1bit";
    }
    if (mode == QueryMode::Full2BitBlockwise) {
        return "full2bit_blockwise";
    }
    return "primary4_plus_residual_rerank";
}

QueryMode parse_query_mode(const string &value) {
    if (value == "full2bit" || value == "full_2bit" || value == "1plus1bit") {
        return QueryMode::Full2Bit;
    }
    if (value == "primary1bit" || value == "primary_1bit" || value == "PRIMARY_1BIT") {
        return QueryMode::Primary1Bit;
    }
    if (value == "progressive1p1bit" || value == "progressive_1p1bit" ||
        value == "PROGRESSIVE_1P1BIT" || value == "progressive") {
        return QueryMode::Progressive1p1Bit;
    }
    if (value == "full2bit_blockwise" || value == "FULL_2BIT_BLOCKWISE" ||
        value == "blockwise_full2bit") {
        return QueryMode::Full2BitBlockwise;
    }
    if (value == "residual4" || value == "residual8" ||
        value == "primary4_residual4" || value == "primary4_residual8") {
        return QueryMode::ResidualRerank;
    }
    throw runtime_error(
        "RABITQ_RERANK_MODE only supports full2bit/primary1bit/progressive1p1bit/full2bit_blockwise/residual4/residual8");
}

QueryMode configured_query_mode() {
    const char *rerank_mode = std::getenv("RABITQ_RERANK_MODE");
    if (rerank_mode != nullptr && rerank_mode[0] != '\0') {
        return parse_query_mode(rerank_mode);
    }
    const char *query_mode = std::getenv("RABITQ_QUERY_MODE");
    if (query_mode != nullptr && query_mode[0] != '\0') {
        return parse_query_mode(query_mode);
    }
    return QueryMode::ResidualRerank;
}

size_t configured_residual_bits(size_t default_value) {
    const char *rerank_mode = std::getenv("RABITQ_RERANK_MODE");
    const char *mode = rerank_mode;
    if (mode == nullptr || mode[0] == '\0') {
        mode = std::getenv("RABITQ_QUERY_MODE");
    }
    if (mode != nullptr) {
        const string value(mode);
        if (value == "full2bit" || value == "full_2bit" || value == "1plus1bit" ||
            value == "primary1bit" || value == "primary_1bit" ||
            value == "progressive1p1bit" || value == "progressive_1p1bit" ||
            value == "progressive" || value == "full2bit_blockwise" ||
            value == "blockwise_full2bit") {
            return 1;
        }
        if (value == "residual4" || value == "primary4_residual4") {
            return 4;
        }
        if (value == "residual8" || value == "primary4_residual8") {
            return 8;
        }
    }
    const char *bits = std::getenv("RABITQ_RESIDUAL_BITS");
    if (bits == nullptr || bits[0] == '\0') {
        return default_value;
    }
    if (bits[0] == '-') {
        throw runtime_error("RABITQ_RESIDUAL_BITS must be a non-negative integer");
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(bits, &end, 10);
    if (end == bits) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

void print_run_config(
    const char *dataset_name,
    size_t vecsize,
    size_t qsize,
    size_t vecdim,
    int efConstruction,
    int M,
    int centroid_count,
    size_t rerank_candidates,
    int random_seed,
    size_t residual_bits,
    size_t residual_block_size,
    const char *residual_scale_mode,
    const char *residual_scale_storage,
    QueryMode query_mode,
    const char *path_index,
    const char *path_data,
    const char *path_q,
    const char *path_gt,
    bool external_residual_storage) {
    cout << "Run config:\n";
    cout << "  dataset=" << dataset_name << "\n";
    cout << "  base_count=" << vecsize << "\n";
    cout << "  query_count=" << qsize << "\n";
    cout << "  dimension=" << vecdim << "\n";
    cout << "  M=" << M << " efConstruction=" << efConstruction << "\n";
    const bool split_1p1bit_mode =
        query_mode == QueryMode::Full2Bit ||
        query_mode == QueryMode::Primary1Bit ||
        query_mode == QueryMode::Progressive1p1Bit ||
        query_mode == QueryMode::Full2BitBlockwise;
    cout << "  quantizer="
         << (split_1p1bit_mode
                ? "split 1+1-bit full 2-bit ExRaBitQ"
                : "4-bit ExRaBitQ")
         << " centroid_count=" << centroid_count
         << " random_seed=" << random_seed << "\n";
    cout << "  rerank_mode=" << query_mode_name(query_mode)
         << " method=" << query_mode_name(query_mode)
         << " rerank_candidates=" << rerank_candidates << "\n";
    cout << "  build_distance=float32_l2"
         << " stored_data="
         << (split_1p1bit_mode
                ? "split_1plus1bit_full2bit"
                : (external_residual_storage ? "4bit_rabitq_plus_disk_residual" : "4bit_rabitq_plus_residual"))
         << " residual_bits=" << residual_bits
         << " residual_block_size=" << residual_block_size
         << " residual_scale_mode=" << residual_scale_mode
         << " residual_scale_storage=" << residual_scale_storage
         << " query_distance=" << query_mode_name(query_mode) << "\n";
    cout << "  enabled_paths=full2bit,primary4_residual4,primary4_residual8"
         << " routing="
         << (split_1p1bit_mode ? query_mode_name(query_mode) : "primary4")
         << " rerank_distance="
         << (split_1p1bit_mode
                ? "none"
                : ("residual" + to_string(residual_bits)))
         << "\n";
    cout << "  base_path=" << path_data << "\n";
    cout << "  query_path=" << path_q << "\n";
    cout << "  gt_path=" << path_gt << "\n";
    cout << "  index_path=" << path_index << "\n";
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

size_t getenv_size_t(const char *name, size_t default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    if (value[0] == '-') {
        throw runtime_error(string(name) + " must be a non-negative integer");
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

bool getenv_bool01_strict(const char *name, bool default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    if (std::strcmp(value, "0") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0) {
        return true;
    }
    throw runtime_error(string(name) + " must be 0 or 1");
}

double getenv_double(const char *name, double default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0') {
        throw runtime_error(string(name) + " must be a floating point number");
    }
    return parsed;
}

float getenv_float(const char *name, float default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || !std::isfinite(parsed)) {
        return default_value;
    }
    return parsed;
}

string getenv_string(const char *name, const string &default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return string(value);
}

string join_path(const string &dir, const string &name) {
    if (dir.empty() || dir == ".") {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

void ensure_directory_exists(const string &dir) {
    if (dir.empty() || dir == ".") {
        return;
    }
    string current;
    size_t pos = 0;
    if (dir[0] == '/') {
        current = "/";
        pos = 1;
    }
    while (pos <= dir.size()) {
        const size_t slash = dir.find('/', pos);
        const string part = dir.substr(pos, slash == string::npos ? string::npos : slash - pos);
        if (!part.empty()) {
            if (!current.empty() && current.back() != '/') {
                current += "/";
            }
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                throw runtime_error("cannot create directory: " + current + ": " + strerror(errno));
            }
        }
        if (slash == string::npos) {
            break;
        }
        pos = slash + 1;
    }
}

class DiskPayloadStore {
 public:
    DiskPayloadStore(const string &path, size_t total_bytes)
        : path_(path) {
        fd_ = ::open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
        if (fd_ < 0) {
            throw runtime_error("cannot create payload file: " + path + ": " + strerror(errno));
        }
        if (::ftruncate(fd_, static_cast<off_t>(total_bytes)) != 0) {
            const string error = strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw runtime_error("cannot resize payload file: " + path + ": " + error);
        }
    }

    ~DiskPayloadStore() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const string &path() const {
        return path_;
    }

    void writeRecord(size_t label, const char *record, size_t record_size) {
        size_t written = 0;
        const off_t base_offset = static_cast<off_t>(label * record_size);
        while (written < record_size) {
            const ssize_t n = ::pwrite(
                fd_,
                record + written,
                record_size - written,
                base_offset + static_cast<off_t>(written));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw runtime_error("cannot write payload file: " + path_ + ": " + strerror(errno));
            }
            if (n == 0) {
                throw runtime_error("short write to payload file: " + path_);
            }
            written += static_cast<size_t>(n);
        }
    }

 private:
    string path_;
    int fd_{-1};
};

string quantizer_state_path(const string &index_path) {
    return index_path + ".rabitq";
}

string residual_state_path(const string &index_path) {
    return index_path + ".residual";
}

string residual_state_path(const string &index_path, QueryMode query_mode) {
    (void) query_mode;
    return residual_state_path(index_path);
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

void print_index_file_size(const string &index_path, QueryMode query_mode = QueryMode::ResidualRerank) {
    const string state_path = quantizer_state_path(index_path);
    const string residual_path = residual_state_path(index_path, query_mode);
    const size_t index_bytes = file_size_bytes(index_path);
    const size_t auxiliary_bytes = file_size_bytes(state_path);
    const bool split_1p1bit_mode =
        query_mode == QueryMode::Full2Bit ||
        query_mode == QueryMode::Primary1Bit ||
        query_mode == QueryMode::Progressive1p1Bit ||
        query_mode == QueryMode::Full2BitBlockwise;
    const size_t residual_bytes =
        split_1p1bit_mode ? 0 : file_size_bytes(residual_path);
    const size_t total_bytes = index_bytes + auxiliary_bytes + residual_bytes;
    const double mb = 1000000.0;
    cout << "Index storage size: " << total_bytes / mb << " MB"
         << " (index=" << index_bytes / mb << " MB"
         << ", auxiliary=" << auxiliary_bytes / mb << " MB"
         << ", residual="
         << residual_bytes / mb << " MB"
         << ", total_bytes=" << total_bytes << ")\n";
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

vector<float> train_kmeans_centroids(
    ifstream &input,
    size_t vecdim,
    size_t vecsize,
    size_t centroid_count,
    size_t sample_count,
    int random_seed,
    vector<unsigned char> &scratch) {
    (void) scratch;
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
        read_fvec_as_float(input, samples.data() + i * vecdim, vecdim);
    }

    std::mt19937 rng(random_seed);
    vector<float> min_dist(actual_sample_count, numeric_limits<float>::infinity());
    vector<float> centroids(centroid_count * vecdim, 0.0f);
    vector<size_t> init_ids;
    init_ids.reserve(centroid_count);
    init_ids.push_back(static_cast<size_t>(random_seed) % actual_sample_count);
    copy(
        samples.data() + init_ids[0] * vecdim,
        samples.data() + (init_ids[0] + 1U) * vecdim,
        centroids.data());
    for (size_t centroid_id = 1; centroid_id < centroid_count; ++centroid_id) {
        double objective = 0.0;
        const float *last_centroid = centroids.data() + (centroid_id - 1U) * vecdim;
#pragma omp parallel for reduction(+:objective) schedule(static)
        for (size_t sample_id = 0; sample_id < actual_sample_count; ++sample_id) {
            const float *sample = samples.data() + sample_id * vecdim;
            float dist = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                const float diff = sample[d] - last_centroid[d];
                dist += diff * diff;
            }
            min_dist[sample_id] = std::min(min_dist[sample_id], dist);
            objective += min_dist[sample_id];
        }
        std::uniform_real_distribution<double> pick(0.0, objective);
        double target = pick(rng);
        size_t chosen = actual_sample_count - 1U;
        for (size_t sample_id = 0; sample_id < actual_sample_count; ++sample_id) {
            target -= min_dist[sample_id];
            if (target <= 0.0) {
                chosen = sample_id;
                break;
            }
        }
        init_ids.push_back(chosen);
        copy(
            samples.data() + chosen * vecdim,
            samples.data() + (chosen + 1U) * vecdim,
            centroids.data() + centroid_id * vecdim);
    }

    vector<float> next_centroids(centroid_count * vecdim, 0.0f);
    vector<size_t> counts(centroid_count, 0);
    vector<float> sample_best_dist(actual_sample_count, 0.0f);
    vector<size_t> assignments(actual_sample_count, 0);
    const size_t kmeans_iters = 20;
    const int thread_count = std::max(1, omp_get_max_threads());
    vector<vector<double> > thread_sums(
        static_cast<size_t>(thread_count),
        vector<double>(centroid_count * vecdim, 0.0));
    vector<vector<size_t> > thread_counts(
        static_cast<size_t>(thread_count),
        vector<size_t>(centroid_count, 0));

    for (size_t iter = 0; iter < kmeans_iters; ++iter) {
        fill(next_centroids.begin(), next_centroids.end(), 0.0f);
        fill(counts.begin(), counts.end(), 0);
        for (int tid = 0; tid < thread_count; ++tid) {
            fill(thread_sums[tid].begin(), thread_sums[tid].end(), 0.0);
            fill(thread_counts[tid].begin(), thread_counts[tid].end(), 0);
        }

        double objective = 0.0;
#pragma omp parallel for reduction(+:objective) schedule(static)
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

            assignments[sample_id] = best_centroid;
            sample_best_dist[sample_id] = best_dist;
            objective += best_dist;
            const int tid = omp_get_thread_num();
            ++thread_counts[static_cast<size_t>(tid)][best_centroid];
            double *dst = thread_sums[static_cast<size_t>(tid)].data() + best_centroid * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] += sample[d];
            }
        }

        for (int tid = 0; tid < thread_count; ++tid) {
            for (size_t centroid_id = 0; centroid_id < centroid_count; ++centroid_id) {
                counts[centroid_id] += thread_counts[static_cast<size_t>(tid)][centroid_id];
                float *dst = next_centroids.data() + centroid_id * vecdim;
                const double *src =
                    thread_sums[static_cast<size_t>(tid)].data() + centroid_id * vecdim;
                for (size_t d = 0; d < vecdim; ++d) {
                    dst[d] += static_cast<float>(src[d]);
                }
            }
        }

        size_t empty_count = 0;
        for (size_t centroid_id = 0; centroid_id < centroid_count; ++centroid_id) {
            float *dst = next_centroids.data() + centroid_id * vecdim;
            if (counts[centroid_id] == 0) {
                ++empty_count;
                const size_t fallback_id = static_cast<size_t>(
                    max_element(sample_best_dist.begin(), sample_best_dist.end()) -
                    sample_best_dist.begin());
                const float *fallback = samples.data() + fallback_id * vecdim;
                copy(fallback, fallback + vecdim, dst);
                sample_best_dist[fallback_id] = 0.0f;
                continue;
            }

            const float inv_count = 1.0f / static_cast<float>(counts[centroid_id]);
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] *= inv_count;
            }
        }

        double movement = 0.0;
        for (size_t i = 0; i < centroids.size(); ++i) {
            const double diff = static_cast<double>(centroids[i]) - static_cast<double>(next_centroids[i]);
            movement += diff * diff;
        }
        cout << "kmeans_iter=" << iter
             << " objective=" << objective
             << " movement=" << std::sqrt(movement)
             << " empty_clusters=" << empty_count << "\n";
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
    double graph_total_us_per_query{0.0};
    double graph_prepare_us_per_query{0.0};
    double graph_entry_us_per_query{0.0};
    double graph_base_layer_us_per_query{0.0};
    double graph_distance_us_per_query{0.0};
    double graph_heap_us_per_query{0.0};
    double graph_finalize_residual_us_per_query{0.0};
    double graph_result_sort_us_per_query{0.0};
    double graph_other_us_per_query{0.0};
    size_t actual_search_ef{0};
    size_t rerank_candidates{0};
    double visited_nodes_per_query{0.0};
    double primary_distance_computations_per_query{0.0};
    double full_2bit_distance_computations_per_query{0.0};
    double secondary_code_reads_per_query{0.0};
    double secondary_read_ratio{0.0};
    double secondary_node_refine_rate{0.0};
    double secondary_byte_fraction{0.0};
    double double_compute_rate{0.0};
    double primary_bytes_read_per_query{0.0};
    double secondary_bytes_read_per_query{0.0};
    double total_code_bytes_read_per_query{0.0};
    double avg_bits_read_per_visited_node{0.0};
    double avg_bits_read_per_distance_candidate{0.0};
    double safe_insert_accept_per_query{0.0};
    double safe_insert_reject_per_query{0.0};
    double insert_refine_per_query{0.0};
    double stop_refine_per_query{0.0};
    double result_eviction_refine_per_query{0.0};
    double final_rerank_refine_per_query{0.0};
    double stop_safe_per_query{0.0};
    double continue_safe_per_query{0.0};
    double full2bit_blocks_processed_per_query{0.0};
    double full2bit_dimensions_processed_per_query{0.0};
    double blockwise_candidates_per_query{0.0};
    double blockwise_early_terminated_per_query{0.0};
    double blockwise_early_termination_rate{0.0};
    double avg_processed_blocks_per_blockwise_candidate{0.0};
    double avg_processed_dimensions_per_blockwise_candidate{0.0};
};

static SearchReport test_approx(
    float *massQ,
    size_t qsize,
    RaBitQHierarchicalNSW &appr_alg,
    const string &base_path,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    size_t actual_search_ef,
    size_t rerank_candidates,
    QueryMode query_mode) {
    (void) base_path;
    size_t correct = 0;
    size_t total = 0;
    double hnsw_us = 0.0;
    hnswlib::HierarchicalNSW<float>::SearchProfile profile_sum;

    for (size_t i = 0; i < qsize; i++) {
        StopW hnsw_timer;
        vector<pair<float, labeltype>> results;
        try {
            if (query_mode == QueryMode::Full2Bit) {
                auto queue = appr_alg.searchKnn(massQ + vecdim * i, k);
                results.reserve(queue.size());
                while (!queue.empty()) {
                    results.push_back(queue.top());
                    queue.pop();
                }
                std::reverse(results.begin(), results.end());
            } else if (query_mode == QueryMode::Primary1Bit) {
                hnswlib::HierarchicalNSW<float>::SearchProfile profile;
                auto queue = appr_alg.searchKnnPrimary1Bit(massQ + vecdim * i, k, &profile);
                profile_sum.visited_nodes += profile.visited_nodes;
                profile_sum.primary_distance_computations += profile.primary_distance_computations;
                profile_sum.full_2bit_distance_computations += profile.full_2bit_distance_computations;
                profile_sum.secondary_code_reads += profile.secondary_code_reads;
                profile_sum.primary_payload_bytes_read += profile.primary_payload_bytes_read;
                profile_sum.secondary_payload_bytes_read += profile.secondary_payload_bytes_read;
                results.reserve(queue.size());
                while (!queue.empty()) {
                    results.push_back(queue.top());
                    queue.pop();
                }
                std::reverse(results.begin(), results.end());
            } else if (query_mode == QueryMode::Progressive1p1Bit) {
                hnswlib::HierarchicalNSW<float>::SearchProfile profile;
                const bool force_refine_all = []() {
                    const char *value = std::getenv("RABITQ_PROGRESSIVE_FORCE_REFINE_ALL");
                    return value != nullptr && std::strcmp(value, "1") == 0;
                }();
                const bool refine_only_final = []() {
                    const char *value = std::getenv("RABITQ_PROGRESSIVE_REFINE_ONLY_FINAL");
                    return value != nullptr && std::strcmp(value, "1") == 0;
                }();
                const double progressive_gap_threshold =
                    getenv_double("PROGRESSIVE_GAP_THRESHOLD", -1.0);
                auto queue = appr_alg.searchKnnProgressive1p1Bit(
                    massQ + vecdim * i,
                    k,
                    &profile,
                    force_refine_all,
                    refine_only_final,
                    progressive_gap_threshold);
                profile_sum.visited_nodes += profile.visited_nodes;
                profile_sum.primary_distance_computations += profile.primary_distance_computations;
                profile_sum.full_2bit_distance_computations += profile.full_2bit_distance_computations;
                profile_sum.secondary_code_reads += profile.secondary_code_reads;
                profile_sum.primary_payload_bytes_read += profile.primary_payload_bytes_read;
                profile_sum.secondary_payload_bytes_read += profile.secondary_payload_bytes_read;
                profile_sum.safe_insert_accept_count += profile.safe_insert_accept_count;
                profile_sum.safe_insert_reject_count += profile.safe_insert_reject_count;
                profile_sum.insert_refine_count += profile.insert_refine_count;
                profile_sum.stop_refine_count += profile.stop_refine_count;
                profile_sum.result_eviction_refine_count += profile.result_eviction_refine_count;
                profile_sum.final_rerank_refine_count += profile.final_rerank_refine_count;
                profile_sum.stop_safe_count += profile.stop_safe_count;
                profile_sum.continue_safe_count += profile.continue_safe_count;
                results.reserve(queue.size());
                while (!queue.empty()) {
                    results.push_back(queue.top());
                    queue.pop();
                }
                std::reverse(results.begin(), results.end());
            } else if (query_mode == QueryMode::Full2BitBlockwise) {
                hnswlib::HierarchicalNSW<float>::SearchProfile profile;
                const size_t block_size = getenv_size_t("RABITQ_FULL2BIT_BLOCK_SIZE", 128);
                auto queue = appr_alg.searchKnnFull2BitBlockwise(
                    massQ + vecdim * i,
                    k,
                    block_size,
                    &profile);
                profile_sum.visited_nodes += profile.visited_nodes;
                profile_sum.expanded_nodes += profile.expanded_nodes;
                profile_sum.distance_computations += profile.distance_computations;
                profile_sum.full_2bit_distance_computations += profile.full_2bit_distance_computations;
                profile_sum.payload_bytes_read += profile.payload_bytes_read;
                profile_sum.full2bit_blocks_processed += profile.full2bit_blocks_processed;
                profile_sum.full2bit_dimensions_processed += profile.full2bit_dimensions_processed;
                profile_sum.blockwise_candidates += profile.blockwise_candidates;
                profile_sum.blockwise_early_terminated += profile.blockwise_early_terminated;
                profile_sum.heap_pushes += profile.heap_pushes;
                profile_sum.heap_pops += profile.heap_pops;
                results.reserve(queue.size());
                while (!queue.empty()) {
                    results.push_back(queue.top());
                    queue.pop();
                }
                std::reverse(results.begin(), results.end());
            } else {
                results = appr_alg.searchKnnPlainThenResidualRerankCloserFirst(
                    massQ + vecdim * i,
                    k,
                    rerank_candidates);
            }
        } catch (const std::exception &error) {
            cerr << query_mode_name(query_mode) << "_query_failed"
                 << " query=" << i
                 << " efSearch=" << actual_search_ef
                 << " rerank_candidates=" << rerank_candidates
                 << " error=" << error.what()
                 << "\n";
            throw;
        }
        hnsw_us += hnsw_timer.getElapsedTimeMicro();

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
    report.redundant_rerank_us_per_query = 0.0f;
    report.total_us_per_query = report.hnsw_search_us_per_query;
    report.graph_total_us_per_query = report.hnsw_search_us_per_query;
    report.graph_prepare_us_per_query = 0.0;
    report.graph_entry_us_per_query = 0.0;
    report.graph_base_layer_us_per_query = 0.0;
    report.graph_distance_us_per_query = 0.0;
    report.graph_heap_us_per_query = 0.0;
    report.graph_finalize_residual_us_per_query = 0.0;
    report.graph_result_sort_us_per_query = 0.0;
    const double measured_graph_parts =
        report.graph_prepare_us_per_query +
        report.graph_entry_us_per_query +
        report.graph_base_layer_us_per_query +
        report.graph_finalize_residual_us_per_query +
        report.graph_result_sort_us_per_query;
    report.graph_other_us_per_query =
        std::max(0.0, report.graph_total_us_per_query - measured_graph_parts);
    report.actual_search_ef = actual_search_ef;
    report.rerank_candidates = rerank_candidates;
    if (query_mode == QueryMode::Primary1Bit ||
        query_mode == QueryMode::Progressive1p1Bit ||
        query_mode == QueryMode::Full2BitBlockwise) {
        const double queries = static_cast<double>(qsize);
        report.visited_nodes_per_query = static_cast<double>(profile_sum.visited_nodes) / queries;
        report.primary_distance_computations_per_query =
            static_cast<double>(profile_sum.primary_distance_computations) / queries;
        report.full_2bit_distance_computations_per_query =
            static_cast<double>(profile_sum.full_2bit_distance_computations) / queries;
        report.secondary_code_reads_per_query =
            static_cast<double>(profile_sum.secondary_code_reads) / queries;
        report.secondary_node_refine_rate = profile_sum.primary_distance_computations == 0
            ? 0.0
            : static_cast<double>(profile_sum.secondary_code_reads) /
                  static_cast<double>(profile_sum.primary_distance_computations);
        report.double_compute_rate = profile_sum.primary_distance_computations == 0
            ? 0.0
            : static_cast<double>(profile_sum.full_2bit_distance_computations) /
                  static_cast<double>(profile_sum.primary_distance_computations);
        report.primary_bytes_read_per_query =
            static_cast<double>(profile_sum.primary_payload_bytes_read) / queries;
        report.secondary_bytes_read_per_query =
            static_cast<double>(profile_sum.secondary_payload_bytes_read) / queries;
        const double total_code_bytes =
            static_cast<double>(profile_sum.primary_payload_bytes_read + profile_sum.secondary_payload_bytes_read);
        report.total_code_bytes_read_per_query = total_code_bytes / queries;
        report.secondary_byte_fraction = total_code_bytes == 0.0
            ? 0.0
            : static_cast<double>(profile_sum.secondary_payload_bytes_read) / total_code_bytes;
        report.secondary_read_ratio = report.secondary_byte_fraction;
        report.safe_insert_accept_per_query =
            static_cast<double>(profile_sum.safe_insert_accept_count) / queries;
        report.safe_insert_reject_per_query =
            static_cast<double>(profile_sum.safe_insert_reject_count) / queries;
        report.insert_refine_per_query =
            static_cast<double>(profile_sum.insert_refine_count) / queries;
        report.stop_refine_per_query =
            static_cast<double>(profile_sum.stop_refine_count) / queries;
        report.result_eviction_refine_per_query =
            static_cast<double>(profile_sum.result_eviction_refine_count) / queries;
        report.final_rerank_refine_per_query =
            static_cast<double>(profile_sum.final_rerank_refine_count) / queries;
        report.stop_safe_per_query =
            static_cast<double>(profile_sum.stop_safe_count) / queries;
        report.continue_safe_per_query =
            static_cast<double>(profile_sum.continue_safe_count) / queries;
        if (query_mode == QueryMode::Full2BitBlockwise) {
            report.full_2bit_distance_computations_per_query =
                static_cast<double>(profile_sum.full_2bit_distance_computations) / queries;
            report.total_code_bytes_read_per_query =
                static_cast<double>(profile_sum.payload_bytes_read) / queries;
            report.full2bit_blocks_processed_per_query =
                static_cast<double>(profile_sum.full2bit_blocks_processed) / queries;
            report.full2bit_dimensions_processed_per_query =
                static_cast<double>(profile_sum.full2bit_dimensions_processed) / queries;
            report.blockwise_candidates_per_query =
                static_cast<double>(profile_sum.blockwise_candidates) / queries;
            report.blockwise_early_terminated_per_query =
                static_cast<double>(profile_sum.blockwise_early_terminated) / queries;
            report.blockwise_early_termination_rate = profile_sum.blockwise_candidates == 0
                ? 0.0
                : static_cast<double>(profile_sum.blockwise_early_terminated) /
                      static_cast<double>(profile_sum.blockwise_candidates);
            report.avg_processed_blocks_per_blockwise_candidate = profile_sum.blockwise_candidates == 0
                ? 0.0
                : static_cast<double>(profile_sum.full2bit_blocks_processed) /
                      static_cast<double>(profile_sum.blockwise_candidates);
            report.avg_processed_dimensions_per_blockwise_candidate = profile_sum.blockwise_candidates == 0
                ? 0.0
                : static_cast<double>(profile_sum.full2bit_dimensions_processed) /
                      static_cast<double>(profile_sum.blockwise_candidates);
        }
        report.avg_bits_read_per_visited_node = profile_sum.visited_nodes == 0
            ? 0.0
            : 8.0 * report.total_code_bytes_read_per_query * queries /
                  static_cast<double>(profile_sum.visited_nodes);
        report.avg_bits_read_per_distance_candidate = profile_sum.distance_computations == 0
            ? 0.0
            : 8.0 * report.total_code_bytes_read_per_query * queries /
                  static_cast<double>(profile_sum.distance_computations);
    }
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
    size_t rerank_candidates,
    QueryMode query_mode) {
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
        const size_t actual_search_ef = ef;
        const size_t actual_rerank_candidates = std::min(ef, rerank_candidates);
        appr_alg.setEf(actual_search_ef);
        SearchReport report = test_approx(
            massQ,
            qsize,
            appr_alg,
            base_path,
            vecdim,
            answers,
            k,
            actual_search_ef,
            actual_rerank_candidates,
            query_mode);
        report.actual_search_ef = actual_search_ef;
        report.rerank_candidates = actual_rerank_candidates;

        const char *slowest_stage = "prepare";
        double slowest_us = report.graph_prepare_us_per_query;
        auto consider_slowest = [&](const char *stage, double us) {
            if (us > slowest_us) {
                slowest_stage = stage;
                slowest_us = us;
            }
        };
        consider_slowest("entry_search", report.graph_entry_us_per_query);
        consider_slowest("base_layer", report.graph_base_layer_us_per_query);
        consider_slowest("finalize_residual", report.graph_finalize_residual_us_per_query);
        consider_slowest("result_sort", report.graph_result_sort_us_per_query);
        consider_slowest("other", report.graph_other_us_per_query);

        cout << ef << "\t" << report.recall
             << "\t" << report.total_us_per_query << " us"
             << "\t" << "method=" << query_mode_name(query_mode)
             << "\t" << "recall_at=" << k
             << "\t" << "requested_ef=" << ef
             << "\t" << "actual_search_ef=" << report.actual_search_ef
             << "\t" << "rerank_candidates=" << report.rerank_candidates;
        cout << "\t" << "hnsw_search_us_per_query=" << report.hnsw_search_us_per_query
             << "\t" << "total_us_per_query=" << report.total_us_per_query
             << "\t" << "graph_total_us_per_query=" << report.graph_total_us_per_query
             << "\t" << "graph_prepare_us_per_query=" << report.graph_prepare_us_per_query
             << "\t" << "graph_entry_search_us_per_query=" << report.graph_entry_us_per_query
             << "\t" << "graph_base_layer_us_per_query=" << report.graph_base_layer_us_per_query
             << "\t" << "graph_distance_us_per_query=" << report.graph_distance_us_per_query
             << "\t" << "graph_heap_us_per_query=" << report.graph_heap_us_per_query
             << "\t" << "graph_finalize_residual_us_per_query="
             << report.graph_finalize_residual_us_per_query
             << "\t" << "graph_result_sort_us_per_query=" << report.graph_result_sort_us_per_query
             << "\t" << "graph_other_us_per_query=" << report.graph_other_us_per_query
             << "\t" << "graph_slowest_stage=" << slowest_stage
             << "\t" << "graph_slowest_us_per_query=" << slowest_us;
        if (query_mode == QueryMode::Primary1Bit ||
            query_mode == QueryMode::Progressive1p1Bit ||
            query_mode == QueryMode::Full2BitBlockwise) {
            cout << "\t" << "visited_nodes_per_query=" << report.visited_nodes_per_query
                 << "\t" << "primary_distance_computations_per_query="
                 << report.primary_distance_computations_per_query
                 << "\t" << "full_2bit_distance_computations_per_query="
                 << report.full_2bit_distance_computations_per_query
                 << "\t" << "secondary_code_reads_per_query=" << report.secondary_code_reads_per_query
                 << "\t" << "secondary_node_refine_rate=" << report.secondary_node_refine_rate
                 << "\t" << "secondary_byte_fraction=" << report.secondary_byte_fraction
                 << "\t" << "double_compute_rate=" << report.double_compute_rate
                 << "\t" << "secondary_read_ratio=" << report.secondary_read_ratio
                 << "\t" << "estimated_primary_bytes_per_query=" << report.primary_bytes_read_per_query
                 << "\t" << "estimated_secondary_bytes_per_query=" << report.secondary_bytes_read_per_query
                 << "\t" << "estimated_total_code_bytes_per_query="
                 << report.total_code_bytes_read_per_query
                 << "\t" << "avg_bits_read_per_visited_node="
                 << report.avg_bits_read_per_visited_node
                 << "\t" << "avg_bits_read_per_distance_candidate="
                 << report.avg_bits_read_per_distance_candidate
                 << "\t" << "safe_insert_accept_per_query=" << report.safe_insert_accept_per_query
                 << "\t" << "safe_insert_reject_per_query=" << report.safe_insert_reject_per_query
                 << "\t" << "insert_refine_per_query=" << report.insert_refine_per_query
                 << "\t" << "stop_refine_per_query=" << report.stop_refine_per_query
                 << "\t" << "result_eviction_refine_per_query="
                 << report.result_eviction_refine_per_query
                 << "\t" << "final_rerank_refine_per_query="
                 << report.final_rerank_refine_per_query
                 << "\t" << "stop_safe_per_query=" << report.stop_safe_per_query
                 << "\t" << "continue_safe_per_query=" << report.continue_safe_per_query
                 << "\t" << "full2bit_blocks_processed_per_query="
                 << report.full2bit_blocks_processed_per_query
                 << "\t" << "full2bit_dimensions_processed_per_query="
                 << report.full2bit_dimensions_processed_per_query
                 << "\t" << "blockwise_candidates_per_query="
                 << report.blockwise_candidates_per_query
                 << "\t" << "blockwise_early_terminated_per_query="
                 << report.blockwise_early_terminated_per_query
                 << "\t" << "blockwise_early_termination_rate="
                 << report.blockwise_early_termination_rate
                 << "\t" << "avg_processed_blocks_per_blockwise_candidate="
                 << report.avg_processed_blocks_per_blockwise_candidate
                 << "\t" << "avg_processed_dimensions_per_blockwise_candidate="
                 << report.avg_processed_dimensions_per_blockwise_candidate;
        }
        cout << "\n";
        if (report.recall > 1.0f) {
            cout << report.recall << "\t" << report.total_us_per_query << " us\n";
            break;
        }
    }
}

void sift_test1B() {
    const int efConstruction = static_cast<int>(getenv_size_t("RABITQ_EF_CONSTRUCTION", 400));
    const int M = static_cast<int>(getenv_size_t("RABITQ_M", 32));
    const int centroid_count = static_cast<int>(getenv_size_t("RABITQ_CENTROID_COUNT", 256));
    const size_t rerank_candidates =
        getenv_size_t("RABITQ_RERANK_CANDIDATES", 100);
    const int random_seed = 100;

    struct DatasetConfig {
        string name;
        size_t dim;
        size_t gt_width;
        string base_path;
        string query_path;
        string gt_path;
        string index_prefix;
    };

    const string dataset_name_config = getenv_string("RABITQ_DATASET", "sift10m");
    const DatasetConfig dataset{
        dataset_name_config,
        getenv_size_t("RABITQ_DIM", 128),
        getenv_size_t("RABITQ_GT_WIDTH", 1000),
        getenv_string(
            "RABITQ_BASE_PATH",
            "/home/kai3/coco/data/sift10m/sift10m_base.fvecs"),
        getenv_string(
            "RABITQ_QUERY_PATH",
            "/home/kai3/coco/data/sift10m/sift10m_query.fvecs"),
        getenv_string(
            "RABITQ_GT_PATH",
            "/home/kai3/coco/data/sift10m/sift10m_groundtruth.ivecs"),
        dataset_name_config};

    const char *dataset_name = dataset.name.c_str();
    const size_t vecdim = dataset.dim;
    const size_t gt_width = dataset.gt_width;
    const QueryMode query_mode = configured_query_mode();
    const bool split_1p1bit_mode =
        query_mode == QueryMode::Full2Bit ||
        query_mode == QueryMode::Primary1Bit ||
        query_mode == QueryMode::Progressive1p1Bit ||
        query_mode == QueryMode::Full2BitBlockwise;
    const bool external_residual_storage = !split_1p1bit_mode;
    const size_t residual_bits = configured_residual_bits(4);
    if (split_1p1bit_mode) {
        if (residual_bits != 1) {
            throw runtime_error("1+1-bit modes use split 1-bit primary + 1-bit residual");
        }
    } else if (residual_bits != 4 && residual_bits != 8) {
        throw runtime_error("Only full2bit, primary1bit, progressive1p1bit, primary4+residual4 and primary4+residual8 are supported");
    }
    const size_t default_centroid_train_samples = vecdim == 128 && dataset.name == "sift10m"
        ? 10000000
        : fvec_count_from_file_size(dataset.base_path.c_str(), vecdim);
    const size_t centroid_train_samples =
        getenv_size_t("RABITQ_CENTROID_TRAIN_SAMPLES", default_centroid_train_samples);
    const hnswlib::RaBitQSpace::ResidualQuantizationConfig residual_config =
        [&]() {
            hnswlib::RaBitQSpace::ResidualQuantizationConfig config;
            config.full2bit_baseline = split_1p1bit_mode;
            config.bits = split_1p1bit_mode
                ? hnswlib::RaBitQSpace::ResidualQuantizationBits::B1
                : (residual_bits == 4
                ? hnswlib::RaBitQSpace::ResidualQuantizationBits::B4
                : hnswlib::RaBitQSpace::ResidualQuantizationBits::B8);
            config.block_size = getenv_size_t("RABITQ_RESIDUAL_BLOCK_SIZE", 16);
            if (config.block_size == 0) {
                throw runtime_error("RABITQ_RESIDUAL_BLOCK_SIZE must be a positive integer");
            }
            if (const char *value = std::getenv("RABITQ_RESIDUAL_BLOCK_SIZE")) {
                char *end = nullptr;
                const unsigned long parsed = std::strtoul(value, &end, 10);
                if (end == value || *end != '\0' || parsed == 0) {
                    throw runtime_error("RABITQ_RESIDUAL_BLOCK_SIZE must be a positive integer");
                }
                config.block_size = static_cast<size_t>(parsed);
            }
            config.mse_optimal_scale = !split_1p1bit_mode && residual_bits == 4;
            config.scale_fp16 = !split_1p1bit_mode && residual_bits == 4;
            if (const char *value = std::getenv("RABITQ_RESIDUAL_SCALE_MODE")) {
                if (std::strcmp(value, "max_abs") == 0) {
                    config.mse_optimal_scale = false;
                } else if (std::strcmp(value, "mse") == 0 || std::strcmp(value, "mse_optimal") == 0) {
                    config.mse_optimal_scale = true;
                } else {
                    throw runtime_error("RABITQ_RESIDUAL_SCALE_MODE must be max_abs or mse");
                }
            }
            if (const char *value = std::getenv("RABITQ_RESIDUAL_SCALE_STORAGE")) {
                if (std::strcmp(value, "fp16") == 0) {
                    config.scale_fp16 = true;
                } else if (std::strcmp(value, "fp32") == 0) {
                    config.scale_fp16 = false;
                } else {
                    throw runtime_error("RABITQ_RESIDUAL_SCALE_STORAGE must be fp16 or fp32");
                }
            }
            return config;
        }();
    const char *residual_scale_mode =
        residual_config.mse_optimal_scale ? "mse" : "max_abs";
    const char *residual_scale_storage =
        residual_config.scale_fp16 ? "fp16" : "fp32";

    char index_name[1024];
    const char *path_q = dataset.query_path.c_str();
    const char *path_data = dataset.base_path.c_str();
    const char *path_gt = dataset.gt_path.c_str();
    const size_t vecsize = fvec_count_from_file_size(path_data, vecdim);
    const size_t qsize = fvec_count_from_file_size(path_q, vecdim);
    const char *scale_name = residual_config.mse_optimal_scale ? "mse" : "maxabs";
    const char *scale_storage_name = residual_config.scale_fp16 ? "fp16" : "fp32";
    if (split_1p1bit_mode) {
        snprintf(
            index_name,
            sizeof(index_name),
            "%s_full2bit_trueK%d_floatbuild_ef_%d_M_%d.bin",
            dataset.index_prefix.c_str(),
            centroid_count,
            efConstruction,
            M);
    } else {
        snprintf(
            index_name,
            sizeof(index_name),
            "%s_primary4_residual%zu_trueK%d_%s_%s_floatbuild_ef_%d_M_%d.bin",
            dataset.index_prefix.c_str(),
            residual_bits,
            centroid_count,
            scale_name,
            scale_storage_name,
            efConstruction,
            M);
    }
    const string default_index_dir = split_1p1bit_mode
        ? "build/trueK256_train10m_full2bit"
        : (residual_bits == 8
        ? "build/trueK256_train10m_residual8"
        : "build/trueK256_train10m");
    const string index_dir = getenv_string("RABITQ_INDEX_DIR", default_index_dir);
    ensure_directory_exists(index_dir);
    const string path_index_string = join_path(index_dir, index_name);
    const char *path_index = path_index_string.c_str();

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
        residual_bits,
        residual_config.block_size,
        residual_scale_mode,
        residual_scale_storage,
        query_mode,
        path_index,
        path_data,
        path_q,
        path_gt,
        external_residual_storage);

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
        vecdim,
        vecsize,
        centroid_count,
        M,
        efConstruction,
        random_seed,
        false,
        external_residual_storage,
        residual_config);
    cout << "  encoded_bytes_per_vector=" << appr_alg->space().get_data_size()
         << (split_1p1bit_mode
                ? " (split 1+1-bit planes in index)\n"
                : (external_residual_storage
                ? " (4-bit code in index; residual in mmap sidecar)\n"
                : " (4-bit code + residual code)\n"));
    cout << "  residual_bits=" << appr_alg->space().get_residual_bits()
         << " residual_block_size=" << appr_alg->space().get_residual_block_size()
         << " residual_scale_mode="
         << (appr_alg->space().get_residual_mse_optimal_scale() ? "mse" : "max_abs")
         << " primary_bits=" << (split_1p1bit_mode ? 1 : hnswlib::RaBitQSpace::kTotalBits)
         << " short_bits=" << hnswlib::RaBitQSpace::kShortBits
         << " remaining_bits=" << (split_1p1bit_mode ? 1 : hnswlib::RaBitQSpace::kRemainingBits)
         << " compact_primary_bytes_per_vector=" << appr_alg->space().get_compact_code_bytes()
         << " full2bit_plane_bytes_per_vector=" << appr_alg->space().get_full2bit_plane_bytes()
         << " residual_code_bytes_per_vector=" << appr_alg->space().get_residual_code_bytes()
         << " residual_record_bytes_per_vector=" << appr_alg->space().get_residual_disk_record_bytes()
         << " primary_4bit_empirical_error_reference="
         << appr_alg->space().get_primary_code_empirical_error_reference()
         << "\n";

    bool need_build = true;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        if (!exists_test(quantizer_state_path(path_index))) {
            cout << "Missing RaBitQ quantizer state sidecar; rebuilding the index\n";
        } else {
            try {
                appr_alg->loadIndex(path_index, vecsize);
                print_index_file_size(path_index, query_mode);
                need_build = false;
            } catch (const std::exception &error) {
                cout << "Existing index is incompatible: " << error.what() << "\n";
                cout << "Rebuilding the index with the current quantizer format\n";
                delete appr_alg;
                appr_alg = new RaBitQHierarchicalNSW(
                    vecdim,
                    vecsize,
                    centroid_count,
                    M,
                    efConstruction,
                    random_seed,
                    false,
                    external_residual_storage,
                    residual_config);
                cout << "  encoded_bytes_per_vector=" << appr_alg->space().get_data_size()
                     << (split_1p1bit_mode
                            ? " (split 1+1-bit planes in index)\n"
                            : (external_residual_storage
                            ? " (4-bit code in index; residual in mmap sidecar)\n"
                            : " (4-bit code + residual code)\n"));
                cout << "  residual_bits=" << appr_alg->space().get_residual_bits()
                     << " residual_block_size=" << appr_alg->space().get_residual_block_size()
                     << " residual_scale_mode="
                     << (appr_alg->space().get_residual_mse_optimal_scale() ? "mse" : "max_abs")
                     << " primary_bits=" << (split_1p1bit_mode ? 1 : hnswlib::RaBitQSpace::kTotalBits)
                     << " short_bits=" << hnswlib::RaBitQSpace::kShortBits
                     << " remaining_bits=" << (split_1p1bit_mode ? 1 : hnswlib::RaBitQSpace::kRemainingBits)
                     << " compact_primary_bytes_per_vector=" << appr_alg->space().get_compact_code_bytes()
                     << " full2bit_plane_bytes_per_vector=" << appr_alg->space().get_full2bit_plane_bytes()
                     << " residual_code_bytes_per_vector=" << appr_alg->space().get_residual_code_bytes()
                     << " residual_record_bytes_per_vector=" << appr_alg->space().get_residual_disk_record_bytes()
                     << " primary_4bit_empirical_error_reference="
                     << appr_alg->space().get_primary_code_empirical_error_reference()
                     << "\n";
                input.clear();
                input.seekg(0, ios::beg);
            }
        }
    }

    if (need_build) {
        StopW total_build_timer;
        cout << "Building index:\n";
        cout << "Training " << centroid_count << " ExRaBitQ centroid(s) from "
             << min(centroid_train_samples, vecsize) << " base vectors\n";
        StopW train_center_timer;
        vector<float> centroids;
        vector<unsigned char> centroid_scratch;
        if (centroid_count == 1) {
            centroids = train_global_center(
                input,
                vecdim,
                vecsize,
                centroid_train_samples);
            appr_alg->space().setGlobalCenter(centroids.data());
        } else {
            centroids = train_kmeans_centroids(
                input,
                vecdim,
                vecsize,
                static_cast<size_t>(centroid_count),
                centroid_train_samples,
                random_seed,
                centroid_scratch);
            appr_alg->space().setCentroids(centroids.data(), static_cast<size_t>(centroid_count));
        }
        const double train_center_us = train_center_timer.getElapsedTimeMicro();
        cout << "build_stage=train_center"
             << " us=" << train_center_us << "\n";
        vector<size_t> centroid_counts(static_cast<size_t>(centroid_count), 0);

        int j1 = 0;
        StopW stopw;
        StopW float_graph_timer;
        const size_t report_every = 100000;
        const size_t payload_record_size = external_residual_storage
            ? appr_alg->space().get_full_data_size()
            : appr_alg->space().get_data_size();
        const size_t payload_total_bytes = vecsize * payload_record_size;
        const bool payload_disk_mode = true;
        const string payload_path = string(path_index) + ".payload.tmp";
        vector<char> payloads;
        unique_ptr<DiskPayloadStore> disk_payload;
        if (payload_disk_mode) {
            disk_payload.reset(new DiskPayloadStore(payload_path, payload_total_bytes));
        } else {
            payloads.assign(payload_total_bytes, 0);
        }
        double payload_encode_cpu_us = 0.0;
        cout << "payload_mode=" << (payload_disk_mode ? "disk_file" : "memory_array")
             << " payload_record_bytes=" << payload_record_size
             << " payload_bytes=" << payload_total_bytes;
        if (payload_disk_mode) {
            cout << " payload_path=" << payload_path;
        }
        cout << "\n";

        cout << "Building HNSW graph with float32 L2 distances, then encoding payloads to "
             << (split_1p1bit_mode ? "split 1+1-bit full 2-bit RaBitQ\n" : "4-bit RaBitQ\n");
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
            vector<char> encoded_first(payload_record_size, 0);
            const uint8_t centroid_id = appr_alg->space().assignCentroid(first.data());
            ++centroid_counts[centroid_id];
            if (external_residual_storage) {
                appr_alg->space().encodeVectorFullWithCentroid(first.data(), centroid_id, encoded_first.data());
            } else {
                appr_alg->space().encodeVector(first.data(), encoded_first.data());
            }
            if (payload_disk_mode) {
                disk_payload->writeRecord(0, encoded_first.data(), payload_record_size);
            } else {
                std::memcpy(payloads.data(), encoded_first.data(), payload_record_size);
            }
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
            vector<char> encoded_payload(payload_record_size, 0);
            const uint8_t centroid_id = appr_alg->space().assignCentroid(local_mass.data());
#pragma omp atomic
            centroid_counts[centroid_id]++;
            if (external_residual_storage) {
                appr_alg->space().encodeVectorFullWithCentroid(local_mass.data(), centroid_id, encoded_payload.data());
            } else {
                appr_alg->space().encodeVector(local_mass.data(), encoded_payload.data());
            }
            if (payload_disk_mode) {
                disk_payload->writeRecord(static_cast<size_t>(label), encoded_payload.data(), payload_record_size);
            } else {
                std::memcpy(
                    payloads.data() + static_cast<size_t>(label) * payload_record_size,
                    encoded_payload.data(),
                    payload_record_size);
            }
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
        vector<size_t> sorted_centroid_counts = centroid_counts;
        sort(sorted_centroid_counts.begin(), sorted_centroid_counts.end());
        const size_t empty_centroids = static_cast<size_t>(
            count(sorted_centroid_counts.begin(), sorted_centroid_counts.end(), size_t{0}));
        const size_t p50_index = sorted_centroid_counts.empty() ? 0 : sorted_centroid_counts.size() / 2U;
        const size_t p95_index = sorted_centroid_counts.empty() ? 0 :
            min(sorted_centroid_counts.size() - 1U,
                static_cast<size_t>(ceil(0.95 * static_cast<double>(sorted_centroid_counts.size()))) - 1U);
        cout << "centroid_assignment_stats"
             << " count=" << centroid_count
             << " min=" << (sorted_centroid_counts.empty() ? 0 : sorted_centroid_counts.front())
             << " max=" << (sorted_centroid_counts.empty() ? 0 : sorted_centroid_counts.back())
             << " p50=" << (sorted_centroid_counts.empty() ? 0 : sorted_centroid_counts[p50_index])
             << " p95=" << (sorted_centroid_counts.empty() ? 0 : sorted_centroid_counts[p95_index])
             << " empty=" << empty_centroids
             << "\n";

        StopW convertw;
        if (external_residual_storage && payload_disk_mode) {
            appr_alg->importGraphFromFloatIndexWithFullPayloadFileAndExternalResiduals(
                float_index,
                payload_path,
                residual_state_path(path_index, query_mode),
                payload_record_size,
                true);
            disk_payload.reset();
            std::remove(payload_path.c_str());
        } else if (external_residual_storage) {
            const string residual_path = residual_state_path(path_index, query_mode);
            DiskPayloadStore residual_store(
                residual_path,
                vecsize * appr_alg->space().get_residual_disk_record_bytes());
            vector<char> compact_payloads(vecsize * appr_alg->space().get_data_size(), 0);
            vector<char> residual_record(appr_alg->space().get_residual_disk_record_bytes(), 0);
            for (size_t label = 0; label < vecsize; ++label) {
                const char *full = payloads.data() + label * payload_record_size;
                appr_alg->space().copyCompactPayloadFromFull(
                    full,
                    compact_payloads.data() + label * appr_alg->space().get_data_size());
                appr_alg->space().copyResidualRecordFromFull(full, residual_record.data());
                residual_store.writeRecord(
                    label,
                    residual_record.data(),
                    appr_alg->space().get_residual_disk_record_bytes());
            }
            appr_alg->importGraphFromFloatIndexWithPayloads(
                float_index,
                compact_payloads,
                appr_alg->space().get_data_size(),
                true);
            appr_alg->space().openExternalResidualStorage(residual_path, vecsize);
        } else {
            if (payload_disk_mode) {
                appr_alg->importGraphFromFloatIndexWithPayloadFile(
                    float_index,
                    payload_path,
                    payload_record_size,
                    true);
                disk_payload.reset();
                std::remove(payload_path.c_str());
            } else {
                appr_alg->importGraphFromFloatIndexWithPayloads(
                    float_index,
                    payloads,
                    payload_record_size,
                    true);
            }
        }
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
        print_index_file_size(path_index, query_mode);
    }

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    const size_t k = 10;
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
        rerank_candidates,
        query_mode);
    print_index_file_size(path_index, query_mode);
}
