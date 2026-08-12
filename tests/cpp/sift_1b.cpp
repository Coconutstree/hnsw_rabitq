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
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"
#include "../../hnswlib/rabitq_vamana.h"

using namespace std;
using namespace hnswlib;

namespace {

class FvecMmap {
 public:
    FvecMmap(const string &path, size_t count, size_t dim)
        : count_(count), dim_(dim), record_bytes_(sizeof(int32_t) + dim * sizeof(float)) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw runtime_error("cannot mmap base vectors: " + path);
        bytes_ = count_ * record_bytes_;
        mapping_ = static_cast<const char *>(::mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0));
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw runtime_error("mmap failed for base vectors: " + path);
        }
    }

    ~FvecMmap() {
        if (mapping_) ::munmap(const_cast<char *>(mapping_), bytes_);
        if (fd_ >= 0) ::close(fd_);
    }

    const float *vector(size_t label) const {
        if (label >= count_) throw out_of_range("base-vector label outside mmap");
        const char *record = mapping_ + label * record_bytes_;
        int32_t stored_dim = 0;
        std::memcpy(&stored_dim, record, sizeof(stored_dim));
        if (stored_dim != static_cast<int32_t>(dim_))
            throw runtime_error("invalid fvec dimension in mmap");
        return reinterpret_cast<const float *>(record + sizeof(int32_t));
    }

 private:
    int fd_{-1};
    const char *mapping_{nullptr};
    size_t bytes_{0};
    size_t count_{0};
    size_t dim_{0};
    size_t record_bytes_{0};
};

enum class QueryMode {
    PrimaryOnly,
    ResidualRerank,
};

const char *query_mode_name(QueryMode mode) {
    return mode == QueryMode::PrimaryOnly
        ? "primary4_only"
        : "primary4_plus_residual_rerank";
}

QueryMode parse_query_mode(const string &value) {
    if (value == "primary4_only" || value == "primary_only") {
        return QueryMode::PrimaryOnly;
    }
    if (value == "residual4" || value == "residual8" ||
        value == "primary4_residual4" || value == "primary4_residual8") {
        return QueryMode::ResidualRerank;
    }
    throw runtime_error(
        "RABITQ_RERANK_MODE only supports primary4_only/residual4/residual8");
}

QueryMode configured_query_mode() {
    const char *abc = std::getenv("RABITQ_ABC_ABLATION");
    if (abc != nullptr && std::strcmp(abc, "A") == 0)
        return QueryMode::PrimaryOnly;
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
    bool external_residual_storage,
    const string &build_distance) {
    cout << "Run config:\n";
    cout << "  dataset=" << dataset_name << "\n";
    cout << "  base_count=" << vecsize << "\n";
    cout << "  query_count=" << qsize << "\n";
    cout << "  dimension=" << vecdim << "\n";
    cout << "  M=" << M << " efConstruction=" << efConstruction << "\n";
    cout << "  quantizer=4-bit ExRaBitQ centroid_count=" << centroid_count
         << " random_seed=" << random_seed << "\n";
    cout << "  rerank_mode=" << query_mode_name(query_mode)
         << " method=" << query_mode_name(query_mode)
         << " rerank_candidates=" << rerank_candidates << "\n";
    cout << "  build_distance=" << build_distance
         << " stored_data="
         << (external_residual_storage ? "4bit_rabitq_plus_disk_residual" : "4bit_rabitq_plus_residual")
         << " residual_bits=" << residual_bits
         << " residual_block_size=" << residual_block_size
         << " residual_scale_mode=" << residual_scale_mode
         << " residual_scale_storage=" << residual_scale_storage
         << " query_distance=" << query_mode_name(query_mode) << "\n";
    cout << "  enabled_paths=primary4_residual4,primary4_residual8"
         << " routing=primary4 rerank_distance=residual" << residual_bits << "\n";
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
    (void) query_mode;
    const string state_path = quantizer_state_path(index_path);
    const string residual_path = residual_state_path(index_path, query_mode);
    const size_t index_bytes = file_size_bytes(index_path);
    const size_t auxiliary_bytes = file_size_bytes(state_path);
    const size_t residual_bytes = file_size_bytes(residual_path);
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

    std::mt19937 rng(random_seed);
    vector<size_t> sample_ids(vecsize);
    std::iota(sample_ids.begin(), sample_ids.end(), size_t{0});
    for (size_t i = 0; i < actual_sample_count; ++i) {
        std::uniform_int_distribution<size_t> pick_id(i, vecsize - 1U);
        std::swap(sample_ids[i], sample_ids[pick_id(rng)]);
    }
    sample_ids.resize(actual_sample_count);
    std::sort(sample_ids.begin(), sample_ids.end());
    vector<float> samples(actual_sample_count * vecdim, 0.0f);
    const size_t record_bytes = sizeof(int) + vecdim * sizeof(float);
    for (size_t i = 0; i < actual_sample_count; ++i) {
        input.seekg(static_cast<std::streamoff>(sample_ids[i] * record_bytes), ios::beg);
        read_fvec_as_float(input, samples.data() + i * vecdim, vecdim);
    }

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
    size_t sample_count,
    int random_seed) {
    input.clear();
    input.seekg(0, ios::beg);

    const size_t actual_sample_count = min(sample_count, vecsize);
    if (actual_sample_count == 0) {
        throw runtime_error("not enough samples to train global center");
    }

    vector<float> center(vecdim, 0.0f);
    vector<float> sample(vecdim, 0.0f);
    vector<size_t> sample_ids(vecsize);
    std::iota(sample_ids.begin(), sample_ids.end(), size_t{0});
    std::mt19937 rng(random_seed);
    for (size_t i = 0; i < actual_sample_count; ++i) {
        std::uniform_int_distribution<size_t> pick_id(i, vecsize - 1U);
        std::swap(sample_ids[i], sample_ids[pick_id(rng)]);
    }
    sample_ids.resize(actual_sample_count);
    std::sort(sample_ids.begin(), sample_ids.end());
    const size_t record_bytes = sizeof(int) + vecdim * sizeof(float);
    for (size_t i = 0; i < actual_sample_count; ++i) {
        input.seekg(static_cast<std::streamoff>(sample_ids[i] * record_bytes), ios::beg);
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
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double visited_nodes_per_query{0.0};
    double distance_computations_per_query{0.0};
    double active_centroids_per_query{0.0};
    double rerank_us_per_query{0.0};
    long long cache_misses{-1};
    long long dtlb_load_misses{-1};
    double short_checked_per_query{0.0};
    double short_would_reject_per_query{0.0};
    double short_ambiguous_per_query{0.0};
    size_t unsafe_reject_total{0};
    size_t short_bound_violation_total{0};
    double full_distance_count_per_query{0.0};
    double short_time_us_per_query{0.0};
    double full_distance_time_us_per_query{0.0};
    double two_bit_checked_per_query{0.0};
    double two_bit_would_reject_per_query{0.0};
    double two_bit_ambiguous_per_query{0.0};
    size_t two_bit_unsafe_reject_total{0};
    size_t two_bit_bound_violation_total{0};
    double two_bit_time_us_per_query{0.0};
    double paper_checked_per_query{0.0};
    double paper_would_prune_per_query{0.0};
    double paper_not_pruned_per_query{0.0};
    size_t paper_false_prune_against_baseline_total{0};
    size_t paper_pruned_baseline_accept_total{0};
    size_t paper_pruned_baseline_reject_total{0};
    double paper_full_saved_per_query{0.0};
    double paper_short_time_us_per_query{0.0};
    double paper_remaining_time_us_per_query{0.0};
    double paper_msb_kernel_calls_per_query{0.0};
    double paper_remaining_kernel_calls_per_query{0.0};
};

struct HardwareCounters {
    int cache_fd{-1};
    int dtlb_fd{-1};
    HardwareCounters() {
#ifdef __linux__
        perf_event_attr attr{};
        attr.size = sizeof(attr);
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CACHE_MISSES;
        cache_fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
        attr.type = PERF_TYPE_HW_CACHE;
        attr.config = PERF_COUNT_HW_CACHE_DTLB |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        dtlb_fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
#endif
    }
    void start() {
#ifdef __linux__
        for (int fd : {cache_fd, dtlb_fd}) if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
#endif
    }
    std::pair<long long, long long> stop() {
        long long values[2] = {-1, -1};
#ifdef __linux__
        const int fds[2] = {cache_fd, dtlb_fd};
        for (size_t i = 0; i < 2; ++i) if (fds[i] >= 0) {
            ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
            if (::read(fds[i], &values[i], sizeof(values[i])) != sizeof(values[i])) values[i] = -1;
        }
#endif
        return {values[0], values[1]};
    }
    ~HardwareCounters() {
        if (cache_fd >= 0) ::close(cache_fd);
        if (dtlb_fd >= 0) ::close(dtlb_fd);
    }
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
    double prepare_us = 0.0;
    double traversal_us = 0.0;
    double rerank_us = 0.0;
    size_t visited_nodes = 0;
    size_t distance_computations = 0;
    size_t active_centroids = 0;
    size_t short_checked = 0, short_would_reject = 0, short_ambiguous = 0;
    size_t unsafe_reject = 0, short_bound_violation = 0, full_distance_count = 0;
    double short_time_us = 0.0, full_distance_time_us = 0.0;
    size_t two_bit_checked = 0, two_bit_would_reject = 0, two_bit_ambiguous = 0;
    size_t two_bit_unsafe_reject = 0, two_bit_bound_violation = 0;
    double two_bit_time_us = 0.0;
    size_t paper_checked = 0, paper_would_prune = 0, paper_not_pruned = 0;
    size_t paper_false_prune = 0, paper_pruned_accept = 0, paper_pruned_reject = 0;
    size_t paper_full_saved = 0, paper_msb_calls = 0, paper_remaining_calls = 0;
    double paper_short_time_us = 0.0, paper_remaining_time_us = 0.0;
    vector<double> latencies;
    latencies.reserve(qsize);

    HardwareCounters hardware_counters;
    hardware_counters.start();
    for (size_t i = 0; i < qsize; i++) {
        StopW hnsw_timer;
        vector<pair<float, labeltype>> results;
        hnswlib::RaBitQSearchMetrics metrics;
        try {
            if (query_mode == QueryMode::PrimaryOnly) {
                const auto raw = appr_alg.searchKnnPrimaryOnly(
                    massQ + vecdim * i, k, nullptr, &metrics);
                auto copy = raw;
                results.reserve(copy.size());
                while (!copy.empty()) {
                    results.push_back(copy.top());
                    copy.pop();
                }
                std::reverse(results.begin(), results.end());
            } else {
                results = appr_alg.searchKnnPlainThenResidualRerankCloserFirst(
                    massQ + vecdim * i,
                    k,
                    rerank_candidates,
                    nullptr,
                    &metrics);
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
        const double elapsed_us = hnsw_timer.getElapsedTimeMicro();
        hnsw_us += elapsed_us;
        latencies.push_back(elapsed_us);
        prepare_us += metrics.prepare_query_us;
        traversal_us += metrics.traversal_us;
        rerank_us += metrics.rerank_us;
        visited_nodes += metrics.visited_nodes;
        distance_computations += metrics.distance_computations;
        active_centroids += metrics.active_centroids;
        short_checked += metrics.short_checked;
        short_would_reject += metrics.short_would_reject;
        short_ambiguous += metrics.short_ambiguous;
        unsafe_reject += metrics.unsafe_reject;
        short_bound_violation += metrics.short_bound_violation;
        full_distance_count += metrics.full_distance_count;
        short_time_us += metrics.short_time_us;
        full_distance_time_us += metrics.full_distance_time_us;
        two_bit_checked += metrics.two_bit_checked;
        two_bit_would_reject += metrics.two_bit_would_reject;
        two_bit_ambiguous += metrics.two_bit_ambiguous;
        two_bit_unsafe_reject += metrics.two_bit_unsafe_reject;
        two_bit_bound_violation += metrics.two_bit_bound_violation;
        two_bit_time_us += metrics.two_bit_time_us;
        paper_checked += metrics.paper_checked;
        paper_would_prune += metrics.paper_would_prune;
        paper_not_pruned += metrics.paper_not_pruned;
        paper_false_prune += metrics.paper_false_prune_against_baseline;
        paper_pruned_accept += metrics.paper_pruned_baseline_accept;
        paper_pruned_reject += metrics.paper_pruned_baseline_reject;
        paper_full_saved += metrics.paper_full_saved;
        paper_msb_calls += metrics.paper_msb_kernel_calls;
        paper_remaining_calls += metrics.paper_remaining_kernel_calls;
        paper_short_time_us += metrics.paper_short_time_us;
        paper_remaining_time_us += metrics.paper_remaining_time_us;

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

    const auto hardware_values = hardware_counters.stop();
    SearchReport report;
    report.recall = total == 0 ? 0.0f : 1.0f * correct / total;
    report.hnsw_search_us_per_query = static_cast<float>(hnsw_us / static_cast<double>(qsize));
    report.redundant_rerank_us_per_query = 0.0f;
    report.total_us_per_query = report.hnsw_search_us_per_query;
    report.graph_total_us_per_query = report.hnsw_search_us_per_query;
    report.graph_prepare_us_per_query = prepare_us / static_cast<double>(qsize);
    report.graph_entry_us_per_query = 0.0;
    report.graph_base_layer_us_per_query = traversal_us / static_cast<double>(qsize);
    report.graph_distance_us_per_query = 0.0;
    report.graph_heap_us_per_query = 0.0;
    report.graph_finalize_residual_us_per_query = rerank_us / static_cast<double>(qsize);
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
    std::sort(latencies.begin(), latencies.end());
    const auto percentile = [&latencies](double p) {
        if (latencies.empty()) return 0.0;
        const size_t index = std::min(
            latencies.size() - 1,
            static_cast<size_t>(std::ceil(p * latencies.size())) - 1U);
        return latencies[index];
    };
    report.p50_us = percentile(0.50);
    report.p95_us = percentile(0.95);
    report.p99_us = percentile(0.99);
    report.visited_nodes_per_query = static_cast<double>(visited_nodes) / qsize;
    report.distance_computations_per_query = static_cast<double>(distance_computations) / qsize;
    report.active_centroids_per_query = static_cast<double>(active_centroids) / qsize;
    report.rerank_us_per_query = rerank_us / qsize;
    report.cache_misses = hardware_values.first;
    report.dtlb_load_misses = hardware_values.second;
    report.short_checked_per_query = static_cast<double>(short_checked) / qsize;
    report.short_would_reject_per_query = static_cast<double>(short_would_reject) / qsize;
    report.short_ambiguous_per_query = static_cast<double>(short_ambiguous) / qsize;
    report.unsafe_reject_total = unsafe_reject;
    report.short_bound_violation_total = short_bound_violation;
    report.full_distance_count_per_query = static_cast<double>(full_distance_count) / qsize;
    report.short_time_us_per_query = short_time_us / qsize;
    report.full_distance_time_us_per_query = full_distance_time_us / qsize;
    report.two_bit_checked_per_query = static_cast<double>(two_bit_checked) / qsize;
    report.two_bit_would_reject_per_query = static_cast<double>(two_bit_would_reject) / qsize;
    report.two_bit_ambiguous_per_query = static_cast<double>(two_bit_ambiguous) / qsize;
    report.two_bit_unsafe_reject_total = two_bit_unsafe_reject;
    report.two_bit_bound_violation_total = two_bit_bound_violation;
    report.two_bit_time_us_per_query = two_bit_time_us / qsize;
    report.paper_checked_per_query = static_cast<double>(paper_checked) / qsize;
    report.paper_would_prune_per_query = static_cast<double>(paper_would_prune) / qsize;
    report.paper_not_pruned_per_query = static_cast<double>(paper_not_pruned) / qsize;
    report.paper_false_prune_against_baseline_total = paper_false_prune;
    report.paper_pruned_baseline_accept_total = paper_pruned_accept;
    report.paper_pruned_baseline_reject_total = paper_pruned_reject;
    report.paper_full_saved_per_query = static_cast<double>(paper_full_saved) / qsize;
    report.paper_short_time_us_per_query = paper_short_time_us / qsize;
    report.paper_remaining_time_us_per_query = paper_remaining_time_us / qsize;
    report.paper_msb_kernel_calls_per_query = static_cast<double>(paper_msb_calls) / qsize;
    report.paper_remaining_kernel_calls_per_query = static_cast<double>(paper_remaining_calls) / qsize;
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
    QueryMode query_mode,
    const string &layout = "original") {
    vector<size_t> efs;
    if (std::getenv("RABITQ_ABC_ABLATION") != nullptr) {
        for (size_t i = 1; i <= 30; ++i)
            if (i >= k) efs.push_back(i);
        for (size_t i = 40; i <= 100; i += 10)
            if (i >= k) efs.push_back(i);
        for (size_t i = 140; i <= 460; i += 40)
            if (i >= k) efs.push_back(i);
    } else if (std::getenv("RABITQ_SHORT_SHADOW_COMPARE") != nullptr ||
        std::getenv("RABITQ_TWO_BIT_SHADOW_COMPARE") != nullptr ||
        std::getenv("RABITQ_PAPER_PRUNE_COMPARE") != nullptr) {
        efs = {64, 96, 128, 192, 256, 460};
    } else if (getenv_bool01_strict("RABITQ_STRICT_BFS_REORDER", false)) {
        efs = {64, 96, 128, 192, 256, 460};
    } else if (getenv_bool01_strict("RABITQ_STRICT_ABLATION", false)) {
        efs = {32, 64, 96, 128, 192, 256, 320, 460, 640};
    } else {
        for (size_t i = 1; i <= 30; i++) if (i >= k) efs.push_back(i);
        for (size_t i = 40; i <= 100; i += 10) if (i >= k) efs.push_back(i);
        for (size_t i = 140; i <= 460; i += 40) if (i >= k) efs.push_back(i);
    }

    const uint64_t graph_fingerprint = appr_alg.labelGraphFingerprint();
    cout << "layout=" << layout
         << " graph_fingerprint=" << graph_fingerprint
         << " payload_fingerprint=" << appr_alg.payloadFingerprintByLabel()
         << " residual_fingerprint=" << appr_alg.residualFingerprintByLabel()
         << " average_neighbor_id_distance=" << appr_alg.averageLevel0NeighborIdDistance()
         << "\n";
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

        if (std::getenv("RABITQ_ABC_ABLATION") != nullptr) {
            cout << ef << "\t" << report.recall
                 << "\t" << report.total_us_per_query << " us"
                 << "\thnsw_search_us_per_query=" << report.hnsw_search_us_per_query
                 << "\tresidual_rerank_us_per_query=" << report.rerank_us_per_query
                 << "\trerank_candidates=" << report.rerank_candidates
                 << "\ttotal_us_per_query=" << report.total_us_per_query
                 << "\tqps="
                 << (report.total_us_per_query > 0.0 ? 1e6 / report.total_us_per_query : 0.0)
                 << "\tp95_us=" << report.p95_us
                 << "\tvisited_nodes=" << report.visited_nodes_per_query
                 << "\tdistance_computations=" << report.distance_computations_per_query
                 << "\tpaper_prune_ratio="
                 << (report.paper_checked_per_query > 0.0
                        ? report.paper_would_prune_per_query / report.paper_checked_per_query : 0.0)
                 << "\tpaper_saved_ratio="
                 << (report.full_distance_count_per_query > 0.0
                        ? report.paper_full_saved_per_query /
                            (report.full_distance_count_per_query +
                             (appr_alg.getGraphTurboConfig().paper_active
                                ? report.paper_full_saved_per_query : 0.0)) : 0.0)
                 << "\tpaper_false_prune=" << report.paper_false_prune_against_baseline_total
                 << "\n";
            continue;
        }

        cout << ef << "\t" << report.recall
             << "\t" << report.total_us_per_query << " us"
             << "\t" << "layout=" << layout
             << "\t" << "qps="
             << (report.total_us_per_query > 0.0 ? 1e6 / report.total_us_per_query : 0.0)
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
             << "\t" << "p50_us=" << report.p50_us
             << "\t" << "p95_us=" << report.p95_us
             << "\t" << "p99_us=" << report.p99_us
             << "\t" << "visited_nodes=" << report.visited_nodes_per_query
             << "\t" << "distance_computations=" << report.distance_computations_per_query
             << "\t" << "cache_misses=" << report.cache_misses
             << "\t" << "dtlb_load_misses=" << report.dtlb_load_misses
             << "\t" << "active_centroids=" << report.active_centroids_per_query
             << "\t" << "rerank_us_per_query=" << report.rerank_us_per_query
             << "\t" << "short_checked=" << report.short_checked_per_query
             << "\t" << "short_would_reject=" << report.short_would_reject_per_query
             << "\t" << "short_ambiguous=" << report.short_ambiguous_per_query
             << "\t" << "unsafe_reject_total=" << report.unsafe_reject_total
             << "\t" << "short_bound_violation_total=" << report.short_bound_violation_total
             << "\t" << "full_distance_count=" << report.full_distance_count_per_query
             << "\t" << "theoretical_full_saved=" << report.short_would_reject_per_query
             << "\t" << "short_time_us=" << report.short_time_us_per_query
             << "\t" << "full_distance_time_us=" << report.full_distance_time_us_per_query
             << "\t" << "global_saved_ratio="
             << (report.full_distance_count_per_query > 0.0
                    ? report.short_would_reject_per_query / report.full_distance_count_per_query : 0.0)
             << "\t" << "eligible_reject_ratio="
             << (report.short_checked_per_query > 0.0
                    ? report.short_would_reject_per_query / report.short_checked_per_query : 0.0)
             << "\t" << "two_bit_checked=" << report.two_bit_checked_per_query
             << "\t" << "two_bit_would_reject=" << report.two_bit_would_reject_per_query
             << "\t" << "two_bit_ambiguous=" << report.two_bit_ambiguous_per_query
             << "\t" << "two_bit_unsafe_reject_total=" << report.two_bit_unsafe_reject_total
             << "\t" << "two_bit_bound_violation_total=" << report.two_bit_bound_violation_total
             << "\t" << "two_bit_time_us=" << report.two_bit_time_us_per_query
             << "\t" << "two_bit_global_saved_ratio="
             << (report.full_distance_count_per_query > 0.0
                    ? report.two_bit_would_reject_per_query / report.full_distance_count_per_query : 0.0)
             << "\t" << "two_bit_eligible_reject_ratio="
             << (report.two_bit_checked_per_query > 0.0
                    ? report.two_bit_would_reject_per_query / report.two_bit_checked_per_query : 0.0)
             << "\t" << "paper_checked=" << report.paper_checked_per_query
             << "\t" << "paper_would_prune=" << report.paper_would_prune_per_query
             << "\t" << "paper_not_pruned=" << report.paper_not_pruned_per_query
             << "\t" << "paper_false_prune_against_baseline="
             << report.paper_false_prune_against_baseline_total
             << "\t" << "paper_pruned_baseline_accept=" << report.paper_pruned_baseline_accept_total
             << "\t" << "paper_pruned_baseline_reject=" << report.paper_pruned_baseline_reject_total
             << "\t" << "paper_short_time_us=" << report.paper_short_time_us_per_query
             << "\t" << "paper_remaining_time_us=" << report.paper_remaining_time_us_per_query
             << "\t" << "paper_full_saved=" << report.paper_full_saved_per_query
             << "\t" << "paper_prune_ratio="
             << (report.paper_checked_per_query > 0.0
                    ? report.paper_would_prune_per_query / report.paper_checked_per_query : 0.0)
             << "\t" << "paper_saved_ratio="
             << (report.full_distance_count_per_query > 0.0
                    ? report.paper_full_saved_per_query /
                        (report.full_distance_count_per_query +
                         (appr_alg.getGraphTurboConfig().paper_active
                            ? report.paper_full_saved_per_query : 0.0)) : 0.0)
             << "\t" << "paper_false_prune_ratio="
             << (report.paper_would_prune_per_query > 0.0
                    ? static_cast<double>(report.paper_false_prune_against_baseline_total) /
                        (report.paper_would_prune_per_query * qsize) : 0.0)
             << "\t" << "paper_msb_kernel_calls=" << report.paper_msb_kernel_calls_per_query
             << "\t" << "paper_remaining_kernel_calls=" << report.paper_remaining_kernel_calls_per_query
             << "\t" << "graph_slowest_stage=" << slowest_stage
             << "\t" << "graph_slowest_us_per_query=" << slowest_us
             << "\n";
        const string csv_path = getenv_string("RABITQ_CSV_PATH", "");
        if (!csv_path.empty()) {
            const bool write_header = !exists_test(csv_path);
            ofstream csv(csv_path, ios::app);
            if (!csv.is_open()) {
                throw runtime_error("cannot open benchmark CSV: " + csv_path);
            }
            if (write_header) {
                csv << "dataset,layout,centroid_count,centroid_mode,M,efConstruction,ef,k,"
                       "rerank_candidates,recall,qps,avg_latency_us,p50_us,p95_us,p99_us,"
                       "visited_nodes,distance_computations,prepare_query_us,traversal_us,"
                       "rerank_us,total_query_us,active_centroids,bytes_per_vector,"
                       "quantization_mse,average_relative_distance_error,ns_per_distance,"
                       "cycles_per_distance,run_index,graph_fingerprint,index_format_version\n";
            }
            csv << getenv_string("RABITQ_DATASET", "sift10m") << ','
                << layout << ','
                << appr_alg.space().get_centroid_count() << ','
                << getenv_string("RABITQ_CENTROID_MODE", "eager") << ','
                << getenv_size_t("RABITQ_M", 32) << ','
                << getenv_size_t("RABITQ_EF_CONSTRUCTION", 400) << ','
                << ef << ',' << k << ',' << report.rerank_candidates << ','
                << report.recall << ','
                << (report.total_us_per_query > 0.0 ? 1e6 / report.total_us_per_query : 0.0) << ','
                << report.total_us_per_query << ',' << report.p50_us << ','
                << report.p95_us << ',' << report.p99_us << ','
                << report.visited_nodes_per_query << ','
                << report.distance_computations_per_query << ','
                << report.graph_prepare_us_per_query << ','
                << report.graph_base_layer_us_per_query << ','
                << report.rerank_us_per_query << ',' << report.graph_total_us_per_query << ','
                << report.active_centroids_per_query << ',' << appr_alg.space().get_data_size() << ','
                << "unavailable,unavailable,unavailable,unavailable,"
                << getenv_size_t("RABITQ_RUN_INDEX", 0) << ','
                << graph_fingerprint << ','
                << (appr_alg.space().get_code_layout() == RaBitQCodeLayout::Turbo128
                        ? "EXRBTQ40" : "EXRBTQ31") << '\n';
        }
        if (report.recall > 1.0f) {
            cout << report.recall << "\t" << report.total_us_per_query << " us\n";
            break;
        }
    }
}

struct VamanaSearchReport {
    float recall{0.0f};
    double total_us_per_query{0.0};
    double qps{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double avg_visited_nodes{0.0};
    double avg_distance_computations{0.0};
    double avg_hops{0.0};
    double avg_active_centroids{0.0};
    double traversal_us_per_query{0.0};
    double rerank_us_per_query{0.0};
    double prefetch_issued_per_query{0.0};
    size_t rerank_candidates{0};
    double avg_residual_rerank_count{0.0};
    double paper_checked_per_query{0.0};
    double paper_would_prune_per_query{0.0};
    double paper_not_pruned_per_query{0.0};
    double paper_full_saved_per_query{0.0};
    double paper_msb_kernel_calls_per_query{0.0};
    double paper_remaining_kernel_calls_per_query{0.0};
    double paper_short_time_us_per_query{0.0};
    double paper_remaining_time_us_per_query{0.0};
    long long cache_misses{-1};
    long long dtlb_load_misses{-1};
};

struct VamanaStorageSummary {
    size_t index_bytes{0};
    size_t auxiliary_bytes{0};
    size_t residual_bytes{0};
    size_t total_bytes{0};
    double index_mb{0.0};
    double auxiliary_mb{0.0};
    double residual_mb{0.0};
    double total_mb{0.0};
};

static VamanaStorageSummary vamana_storage_summary(const string &index_path) {
    const double mb = 1000000.0;
    VamanaStorageSummary summary;
    summary.index_bytes = file_size_bytes(index_path);
    summary.auxiliary_bytes = file_size_bytes(quantizer_state_path(index_path));
    summary.residual_bytes = file_size_bytes(residual_state_path(index_path));
    summary.total_bytes =
        summary.index_bytes + summary.auxiliary_bytes + summary.residual_bytes;
    summary.index_mb = summary.index_bytes / mb;
    summary.auxiliary_mb = summary.auxiliary_bytes / mb;
    summary.residual_mb = summary.residual_bytes / mb;
    summary.total_mb = summary.total_bytes / mb;
    return summary;
}

static void print_vamana_run_config(
    const char *dataset_name,
    size_t vecsize,
    size_t qsize,
    size_t vecdim,
    int centroid_count,
    int random_seed,
    size_t residual_bits,
    size_t residual_block_size,
    const char *residual_scale_mode,
    const char *residual_scale_storage,
    QueryMode query_mode,
    size_t rerank_candidates,
    size_t refine_passes,
    bool paper_prune_active,
    float paper_epsilon0,
    const char *path_index,
    const char *path_data,
    const char *path_q,
    const char *path_gt) {
    cout << "Run config:\n";
    cout << "  graph=Vamana\n";
    cout << "  payload=ExRaBitQ4\n";
    cout << "  build_distance=ExRaBitQ4_symmetric\n";
    cout << "  query_distance=Float32_to_ExRaBitQ4\n";
    cout << "  dataset=" << dataset_name << "\n";
    cout << "  base_count=" << vecsize << "\n";
    cout << "  query_count=" << qsize << "\n";
    cout << "  dimension=" << vecdim << "\n";
    cout << "  R=" << hnswlib::VamanaIndex::kDefaultR
         << " L_build=" << hnswlib::VamanaIndex::kDefaultLBuild
         << " alpha=" << hnswlib::VamanaIndex::kDefaultAlpha
         << " beam_width=" << hnswlib::VamanaIndex::kDefaultBeamWidth << "\n";
    cout << "  quantizer=4-bit ExRaBitQ centroid_count=" << centroid_count
         << " random_seed=" << random_seed
         << " code_layout=sequential\n";
    cout << "  residual_bits=" << residual_bits
         << " residual_block_size=" << residual_block_size
         << " residual_scale_mode=" << residual_scale_mode
         << " residual_scale_storage=" << residual_scale_storage
         << " residual_rerank=" << (query_mode == QueryMode::ResidualRerank ? 1 : 0)
         << " rerank_candidates=" << rerank_candidates << "\n";
    cout << "  vamana_refine_passes=" << refine_passes << "\n";
    cout << "  vamana_paper_prune_active=" << (paper_prune_active ? 1 : 0)
         << " paper_epsilon0=" << paper_epsilon0 << "\n";
    cout << "  L_search_sweep=10..30,40..100(step10),140..adaptive_until_recall_0.995(step40)\n";
    cout << "  base_path=" << path_data << "\n";
    cout << "  query_path=" << path_q << "\n";
    cout << "  gt_path=" << path_gt << "\n";
    cout << "  index_path=" << path_index << "\n";
}

static void print_vamana_index_storage(
    const string &index_path,
    const hnswlib::RaBitQVamanaIndex &index) {
    const double mb = 1000000.0;
    const size_t graph_bytes = index.graphStorageBytes();
    const size_t payload_bytes = index.payloadStorageBytes();
    const size_t paper_sidecar_bytes = index.paperPruneSidecarBytes();
    const size_t quantizer_bytes = file_size_bytes(quantizer_state_path(index_path));
    const size_t residual_bytes = file_size_bytes(residual_state_path(index_path));
    const VamanaStorageSummary storage = vamana_storage_summary(index_path);
    cout << "index_storage"
         << " graph=Vamana"
         << " payload=ExRaBitQ4"
         << " graph_size_MB=" << graph_bytes / mb
         << " payload_size_MB=" << payload_bytes / mb
         << " paper_msb_sidecar_MB=" << paper_sidecar_bytes / mb
         << " quantizer_size_MB=" << quantizer_bytes / mb
         << " residual_size_MB=" << residual_bytes / mb
         << " total_index_size_MB=" << storage.total_mb
         << " total_bytes=" << storage.total_bytes
         << "\n";
    cout << "Index storage size: " << storage.total_mb << " MB"
         << " (index=" << storage.index_mb << " MB"
         << ", auxiliary=" << storage.auxiliary_mb << " MB"
         << ", residual=" << storage.residual_mb << " MB"
         << ", total_bytes=" << storage.total_bytes << ")\n";
}

static VamanaSearchReport test_approx_vamana(
    float *massQ,
    size_t qsize,
    hnswlib::RaBitQVamanaIndex &index,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    size_t l_search,
    size_t rerank_candidates,
    QueryMode query_mode) {
    size_t correct = 0;
    size_t total = 0;
    double total_us = 0.0;
    size_t visited_nodes = 0;
    size_t distance_computations = 0;
    size_t hops = 0;
    size_t active_centroids = 0;
    double traversal_us = 0.0;
    double rerank_us = 0.0;
    size_t prefetch_issued = 0;
    size_t residual_rerank_count = 0;
    size_t paper_checked = 0;
    size_t paper_would_prune = 0;
    size_t paper_not_pruned = 0;
    size_t paper_full_saved = 0;
    size_t paper_msb_kernel_calls = 0;
    size_t paper_remaining_kernel_calls = 0;
    double paper_short_time_us = 0.0;
    double paper_remaining_time_us = 0.0;
    vector<double> latencies(qsize, 0.0);

    index.setLSearch(l_search);
    if (index.paperPruneActive()) {
        index.preparePaperPruneSidecar();
    }
    HardwareCounters hardware_counters;
    hardware_counters.start();
    const size_t vecdim = index.space().get_dim();
#pragma omp parallel for schedule(dynamic, 16) reduction(+:correct,total,total_us,visited_nodes,distance_computations,hops,active_centroids,traversal_us,rerank_us,prefetch_issued,residual_rerank_count,paper_checked,paper_would_prune,paper_not_pruned,paper_full_saved,paper_msb_kernel_calls,paper_remaining_kernel_calls,paper_short_time_us,paper_remaining_time_us)
    for (int64_t signed_i = 0; signed_i < static_cast<int64_t>(qsize); ++signed_i) {
        const size_t i = static_cast<size_t>(signed_i);
        StopW timer;
        hnswlib::RaBitQSearchMetrics metrics;
        vector<pair<float, labeltype>> results;
        if (query_mode == QueryMode::PrimaryOnly) {
            results = index.searchKnnPrimaryOnlyCloserFirst(
                massQ + i * vecdim, k, nullptr, &metrics);
        } else {
            results = index.searchKnnPlainThenResidualRerankCloserFirst(
                massQ + i * vecdim, k, rerank_candidates, nullptr, &metrics);
        }
        const double elapsed_us = timer.getElapsedTimeMicro();
        total_us += elapsed_us;
        latencies[i] = elapsed_us;
        visited_nodes += metrics.visited_nodes;
        distance_computations += metrics.distance_computations;
        hops += metrics.hops;
        active_centroids += metrics.active_centroids;
        traversal_us += metrics.traversal_us;
        rerank_us += metrics.rerank_us;
        prefetch_issued += metrics.prefetch_issued;
        residual_rerank_count += metrics.full_distance_count;
        paper_checked += metrics.paper_checked;
        paper_would_prune += metrics.paper_would_prune;
        paper_not_pruned += metrics.paper_not_pruned;
        paper_full_saved += metrics.paper_full_saved;
        paper_msb_kernel_calls += metrics.paper_msb_kernel_calls;
        paper_remaining_kernel_calls += metrics.paper_remaining_kernel_calls;
        paper_short_time_us += metrics.paper_short_time_us;
        paper_remaining_time_us += metrics.paper_remaining_time_us;

        std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();
        while (!gt.empty()) {
            g.insert(gt.top().second);
            gt.pop();
        }
        for (const auto &entry : results) {
            if (g.find(entry.second) != g.end()) {
                ++correct;
            }
        }
    }
    const auto hardware_values = hardware_counters.stop();
    std::sort(latencies.begin(), latencies.end());
    const auto percentile = [&latencies](double p) {
        if (latencies.empty()) return 0.0;
        const size_t index = std::min(
            latencies.size() - 1,
            static_cast<size_t>(std::ceil(p * latencies.size())) - 1U);
        return latencies[index];
    };

    VamanaSearchReport report;
    report.recall = total == 0 ? 0.0f : static_cast<float>(correct) / static_cast<float>(total);
    report.total_us_per_query = qsize == 0 ? 0.0 : total_us / static_cast<double>(qsize);
    report.qps = report.total_us_per_query > 0.0 ? 1e6 / report.total_us_per_query : 0.0;
    report.p50_us = percentile(0.50);
    report.p95_us = percentile(0.95);
    report.p99_us = percentile(0.99);
    report.avg_visited_nodes = qsize == 0 ? 0.0 : static_cast<double>(visited_nodes) / qsize;
    report.avg_distance_computations =
        qsize == 0 ? 0.0 : static_cast<double>(distance_computations) / qsize;
    report.avg_hops = qsize == 0 ? 0.0 : static_cast<double>(hops) / qsize;
    report.avg_active_centroids = qsize == 0 ? 0.0 : static_cast<double>(active_centroids) / qsize;
    report.traversal_us_per_query = qsize == 0 ? 0.0 : traversal_us / qsize;
    report.rerank_us_per_query = qsize == 0 ? 0.0 : rerank_us / qsize;
    report.prefetch_issued_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(prefetch_issued) / qsize;
    report.rerank_candidates = rerank_candidates;
    report.avg_residual_rerank_count =
        qsize == 0 ? 0.0 : static_cast<double>(residual_rerank_count) / qsize;
    report.paper_checked_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(paper_checked) / qsize;
    report.paper_would_prune_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(paper_would_prune) / qsize;
    report.paper_not_pruned_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(paper_not_pruned) / qsize;
    report.paper_full_saved_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(paper_full_saved) / qsize;
    report.paper_msb_kernel_calls_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(paper_msb_kernel_calls) / qsize;
    report.paper_remaining_kernel_calls_per_query =
        qsize == 0 ? 0.0 : static_cast<double>(paper_remaining_kernel_calls) / qsize;
    report.paper_short_time_us_per_query = qsize == 0 ? 0.0 : paper_short_time_us / qsize;
    report.paper_remaining_time_us_per_query =
        qsize == 0 ? 0.0 : paper_remaining_time_us / qsize;
    report.cache_misses = hardware_values.first;
    report.dtlb_load_misses = hardware_values.second;
    return report;
}

static size_t next_vamana_l_search(size_t current, size_t node_count) {
    if (current < 140) return std::min<size_t>(140, node_count);
    if (current < 1000) return std::min(current + 40, node_count);
    if (current < 3000) return std::min(current + 100, node_count);
    if (current < 10000) return std::min(current + 500, node_count);
    if (current < 50000) return std::min(current + 5000, node_count);
    if (current < 200000) return std::min(current + 25000, node_count);
    return std::min(current + 100000, node_count);
}

static void test_vs_recall_vamana(
    float *massQ,
    size_t qsize,
    hnswlib::RaBitQVamanaIndex &index,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    const string &index_path,
    double build_time_ms,
    size_t rerank_candidates,
    QueryMode query_mode,
    size_t refine_passes,
    bool paper_prune_active,
    float paper_epsilon0) {
    const vector<size_t> fixed_l_search_values = {
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        40, 50, 60, 70, 80, 90, 100};
    const float recall_target = getenv_float("RABITQ_RECALL_TARGET", 0.995f);
    const size_t requested_l_search_cap =
        getenv_size_t("RABITQ_LSEARCH_MAX", index.index().size());
    const size_t l_search_cap =
        std::max<size_t>(1, std::min(index.index().size(), requested_l_search_cap));
    if (paper_prune_active) {
        index.preparePaperPruneSidecar();
        cout << "paper_experiment=paper_active"
             << " epsilon0=" << paper_epsilon0
             << " msb_remaining_physical_layout=split_msb_sidecar_plus_full_nibble"
             << " paper_msb_sidecar_bytes=" << index.paperPruneSidecarBytes()
             << " residual_sidecar_accessed_by_paper_prune=0\n";
    }
    const auto degree_stats = index.degreeStats();
    const VamanaStorageSummary storage = vamana_storage_summary(index_path);
    cout << "graph_summary"
         << " graph=Vamana"
         << " payload=ExRaBitQ4"
         << " build_distance=ExRaBitQ4_symmetric"
         << " query_distance=Float32_to_ExRaBitQ4"
         << " R=" << hnswlib::VamanaIndex::kDefaultR
         << " L_build=" << hnswlib::VamanaIndex::kDefaultLBuild
         << " alpha=" << hnswlib::VamanaIndex::kDefaultAlpha
         << " beam_width=" << hnswlib::VamanaIndex::kDefaultBeamWidth
         << " start_node=" << index.index().start_node()
         << " graph_fingerprint=" << index.graphFingerprint()
         << " max_degree=" << degree_stats.max_degree
         << " avg_degree=" << degree_stats.avg_degree
         << " recall_target=" << recall_target
         << " L_search_cap=" << l_search_cap
         << " query_mode=" << query_mode_name(query_mode)
         << " residual_rerank=" << (query_mode == QueryMode::ResidualRerank ? 1 : 0)
         << " rerank_candidates=" << rerank_candidates
         << " refine_passes=" << refine_passes
         << " paper_prune_active=" << (paper_prune_active ? 1 : 0)
         << " paper_epsilon0=" << paper_epsilon0
         << "\n";
    print_vamana_index_storage(index_path, index);

    size_t fixed_index = 0;
    size_t l_search = 0;
    size_t last_l_search = 0;
    float best_recall = 0.0f;
    size_t best_l_search = 0;
    while (true) {
        if (fixed_index < fixed_l_search_values.size()) {
            l_search = fixed_l_search_values[fixed_index++];
        } else {
            const size_t next = next_vamana_l_search(l_search, l_search_cap);
            if (next == l_search) {
                break;
            }
            l_search = next;
        }
        if (l_search > l_search_cap) {
            l_search = l_search_cap;
        }
        if (l_search == last_l_search) {
            if (l_search >= l_search_cap) {
                break;
            }
            continue;
        }
        last_l_search = l_search;
        const VamanaSearchReport report =
            test_approx_vamana(
                massQ, qsize, index, answers, k, l_search,
                rerank_candidates, query_mode);
        if (report.recall > best_recall) {
            best_recall = report.recall;
            best_l_search = l_search;
        }
        cout << "L_search=" << l_search
             << "\trecall@" << k << "=" << report.recall
             << "\tQPS=" << report.qps
             << "\t" << report.total_us_per_query << " us"
             << "\ttotal_us_per_query_us=" << report.total_us_per_query
             << "\tp50_us=" << report.p50_us
             << "\tp95_us=" << report.p95_us
             << "\tp99_us=" << report.p99_us
             << "\tgraph=Vamana"
             << "\tpayload=ExRaBitQ4"
             << "\tbuild_distance=ExRaBitQ4_symmetric"
             << "\tquery_distance=Float32_to_ExRaBitQ4"
             << "\tR=" << hnswlib::VamanaIndex::kDefaultR
             << "\tL_build=" << hnswlib::VamanaIndex::kDefaultLBuild
             << "\talpha=" << hnswlib::VamanaIndex::kDefaultAlpha
             << "\tbeam_width=" << hnswlib::VamanaIndex::kDefaultBeamWidth
             << "\trefine_passes=" << refine_passes
             << "\tprefetch_issued=" << report.prefetch_issued_per_query
             << "\tpaper_prune_active=" << (paper_prune_active ? 1 : 0)
             << "\tpaper_epsilon0=" << paper_epsilon0
             << "\tbuild_time_seconds=" << build_time_ms / 1000.0
             << "\tgraph_size_MB=" << index.graphStorageBytes() / 1000000.0
             << "\tpayload_size_MB=" << index.payloadStorageBytes() / 1000000.0
             << "\tpaper_msb_sidecar_MB=" << index.paperPruneSidecarBytes() / 1000000.0
             << "\tresidual_size_MB=" << file_size_bytes(residual_state_path(index_path)) / 1000000.0
             << "\ttotal_index_size_MB=" << storage.total_mb
             << "\tavg_visited_nodes=" << report.avg_visited_nodes
             << "\tavg_distance_computations=" << report.avg_distance_computations
             << "\tavg_hops=" << report.avg_hops
             << "\tavg_active_centroids=" << report.avg_active_centroids
             << "\tresidual_rerank=" << (query_mode == QueryMode::ResidualRerank ? 1 : 0)
             << "\trerank_candidates=" << report.rerank_candidates
             << "\tavg_residual_rerank_count=" << report.avg_residual_rerank_count
             << "\ttraversal_us_per_query_us=" << report.traversal_us_per_query
             << "\tresidual_rerank_us_per_query_us=" << report.rerank_us_per_query
             << "\tpaper_checked=" << report.paper_checked_per_query
             << "\tpaper_would_prune=" << report.paper_would_prune_per_query
             << "\tpaper_not_pruned=" << report.paper_not_pruned_per_query
             << "\tpaper_full_saved=" << report.paper_full_saved_per_query
             << "\tpaper_msb_kernel_calls=" << report.paper_msb_kernel_calls_per_query
             << "\tpaper_remaining_kernel_calls=" << report.paper_remaining_kernel_calls_per_query
             << "\tpaper_short_time_us=" << report.paper_short_time_us_per_query
             << "\tpaper_remaining_time_us=" << report.paper_remaining_time_us_per_query
             << "\tpaper_prune_ratio="
             << (report.paper_checked_per_query > 0.0
                    ? report.paper_would_prune_per_query / report.paper_checked_per_query
                    : 0.0)
             << "\tpaper_saved_ratio="
             << (report.paper_checked_per_query + report.paper_remaining_kernel_calls_per_query > 0.0
                    ? report.paper_full_saved_per_query /
                        (report.paper_checked_per_query +
                         report.paper_remaining_kernel_calls_per_query)
                    : 0.0)
             << "\tpaper_false_prune=0"
             << "\tmax_degree=" << degree_stats.max_degree
             << "\tavg_degree=" << degree_stats.avg_degree
             << "\tcache_misses=" << report.cache_misses
             << "\tdtlb_load_misses=" << report.dtlb_load_misses
             << "\tIndex storage size: " << storage.total_mb << " MB"
             << " (index=" << storage.index_mb << " MB"
             << ", auxiliary=" << storage.auxiliary_mb << " MB"
             << ", residual=" << storage.residual_mb << " MB"
             << ", total_bytes=" << storage.total_bytes << ")"
             << "\n";
        cout.flush();
        if (report.recall >= recall_target) {
            cout << "L_search_sweep_stop"
                 << " reason=recall_target_reached"
                 << " recall_target=" << recall_target
                 << " L_search=" << l_search
                 << " recall@" << k << "=" << report.recall
                 << "\n";
            return;
        }
        if (l_search >= l_search_cap) {
            break;
        }
    }
    cout << "L_search_sweep_stop"
         << " reason=exhausted_L_search"
         << " recall_target=" << recall_target
         << " L_search_cap=" << l_search_cap
         << " best_L_search=" << best_l_search
         << " best_recall@" << k << "=" << best_recall
         << "\n";
}

static void sift_vamana_test() {
    const int centroid_count = 1;
    const int random_seed = 100;
    const QueryMode query_mode = configured_query_mode();
    const size_t rerank_candidates =
        getenv_size_t("RABITQ_RERANK_CANDIDATES", 100);
    const size_t residual_bits = configured_residual_bits(4);
    if (residual_bits != 4 && residual_bits != 8) {
        throw runtime_error("Only primary4 residual configuration bits 4 or 8 are supported");
    }

    struct DatasetConfig {
        string name;
        size_t dim;
        size_t gt_width;
        string base_path;
        string query_path;
        string gt_path;
        string index_prefix;
    };

    const string dbpedia_data_dir = "/home/kai3/coco/data/dbpedia_openai1536";
    const string dataset_name_config = getenv_string(
        "RABITQ_DATASET", "dbpedia_openai1536");
    const bool is_dbpedia_dataset =
        dataset_name_config == "dbpedia_openai1536" ||
        dataset_name_config == "dbpedia" ||
        dataset_name_config == "dbpedia-openai1536";
    const DatasetConfig dataset{
        dataset_name_config,
        getenv_size_t("RABITQ_DIM", is_dbpedia_dataset ? 1536 : 128),
        getenv_size_t("RABITQ_GT_WIDTH", 100),
        getenv_string(
            "RABITQ_BASE_PATH",
            is_dbpedia_dataset
                ? dbpedia_data_dir + "/dbpedia_openai1536_base.fvecs"
                : "/home/kai3/coco/data/sift10m/sift10m_base.fvecs"),
        getenv_string(
            "RABITQ_QUERY_PATH",
            is_dbpedia_dataset
                ? dbpedia_data_dir + "/dbpedia_openai1536_query.fvecs"
                : "/home/kai3/coco/data/sift10m/sift10m_query.fvecs"),
        getenv_string(
            "RABITQ_GT_PATH",
            is_dbpedia_dataset
                ? dbpedia_data_dir + "/dbpedia_openai1536_groundtruth.ivecs"
                : "/home/kai3/coco/data/sift10m/sift10m_groundtruth.ivecs"),
        dataset_name_config};

    const size_t vecdim = dataset.dim;
    const size_t gt_width = dataset.gt_width;
    const bool external_residual_storage = true;
    hnswlib::RaBitQSpace::ResidualQuantizationConfig residual_config;
    residual_config.bits = residual_bits == 4
        ? hnswlib::RaBitQSpace::ResidualQuantizationBits::B4
        : hnswlib::RaBitQSpace::ResidualQuantizationBits::B8;
    residual_config.block_size = getenv_size_t("RABITQ_RESIDUAL_BLOCK_SIZE", 16);
    if (residual_config.block_size == 0) {
        throw runtime_error("RABITQ_RESIDUAL_BLOCK_SIZE must be a positive integer");
    }
    residual_config.mse_optimal_scale = residual_bits == 4;
    residual_config.scale_fp16 = residual_bits == 4;
    if (const char *value = std::getenv("RABITQ_RESIDUAL_SCALE_MODE")) {
        if (std::strcmp(value, "max_abs") == 0) {
            residual_config.mse_optimal_scale = false;
        } else if (std::strcmp(value, "mse") == 0 || std::strcmp(value, "mse_optimal") == 0) {
            residual_config.mse_optimal_scale = true;
        } else {
            throw runtime_error("RABITQ_RESIDUAL_SCALE_MODE must be max_abs or mse");
        }
    }
    if (const char *value = std::getenv("RABITQ_RESIDUAL_SCALE_STORAGE")) {
        if (std::strcmp(value, "fp16") == 0) {
            residual_config.scale_fp16 = true;
        } else if (std::strcmp(value, "fp32") == 0) {
            residual_config.scale_fp16 = false;
        } else {
            throw runtime_error("RABITQ_RESIDUAL_SCALE_STORAGE must be fp16 or fp32");
        }
    }
    const char *residual_scale_mode =
        residual_config.mse_optimal_scale ? "mse" : "max_abs";
    const char *residual_scale_storage =
        residual_config.scale_fp16 ? "fp16" : "fp32";

    const char *path_q = dataset.query_path.c_str();
    const char *path_data = dataset.base_path.c_str();
    const char *path_gt = dataset.gt_path.c_str();
    const size_t full_vecsize = fvec_count_from_file_size(path_data, vecdim);
    const size_t base_limit = getenv_size_t("RABITQ_BASE_LIMIT", 0);
    const size_t vecsize =
        base_limit == 0 ? full_vecsize : std::min(full_vecsize, base_limit);
    const size_t qsize = fvec_count_from_file_size(path_q, vecdim);
    const bool force_rebuild = getenv_bool01_strict("RABITQ_FORCE_REBUILD", false);
    const size_t refine_passes = getenv_size_t("RABITQ_VAMANA_REFINE_PASSES", 1);
    const bool paper_prune_active =
        getenv_bool01_strict("RABITQ_VAMANA_PAPER_PRUNE", true);
    const float paper_epsilon0 =
        getenv_float("RABITQ_VAMANA_PAPER_EPSILON0", 1.9f);
    const size_t default_centroid_train_samples = dataset.name == "sift10m"
        ? 10000000
        : std::min<size_t>(100000, vecsize);
    const size_t centroid_train_samples =
        getenv_size_t("RABITQ_CENTROID_TRAIN_SAMPLES", default_centroid_train_samples);

    const string index_dir = getenv_string("RABITQ_INDEX_DIR", "build/vamana_exrabitq4");
    ensure_directory_exists(index_dir);
    string index_prefix = dataset.index_prefix;
    if (base_limit != 0) {
        index_prefix += "_n_" + std::to_string(vecsize);
    }
    const string path_index_string = join_path(
        index_dir,
        index_prefix +
            "_vamana_exrabitq4_R_32_Lbuild_" +
            std::to_string(hnswlib::VamanaIndex::kDefaultLBuild) +
            "_alpha_1p2_beam_1_refine_" +
            std::to_string(refine_passes) + ".bin");
    const char *path_index = path_index_string.c_str();

    cout << "build_runtime"
         << " omp_max_threads=" << omp_get_max_threads()
         << " omp_dynamic=" << omp_get_dynamic()
         << " full_base_count=" << full_vecsize
         << " effective_base_count=" << vecsize
         << " base_limit=" << base_limit
         << "\n";

    print_vamana_run_config(
        dataset.name.c_str(),
        vecsize,
        qsize,
        vecdim,
        centroid_count,
        random_seed,
        residual_bits,
        residual_config.block_size,
        residual_scale_mode,
        residual_scale_storage,
        query_mode,
        rerank_candidates,
        refine_passes,
        paper_prune_active,
        paper_epsilon0,
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
        if (!inputGT.good() || t != static_cast<int>(gt_width)) {
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

    hnswlib::RaBitQVamanaIndex index(
        vecdim,
        vecsize,
        centroid_count,
        random_seed,
        external_residual_storage,
        residual_config);
    index.space().set_code_layout(hnswlib::RaBitQCodeLayout::SequentialNibble);
    index.setPaperPrune(paper_prune_active, paper_epsilon0);

    const string build_metrics_path = path_index_string + ".build_metrics";
    double build_time_ms = -1.0;
    bool need_build = true;
    if (!force_rebuild && exists_test(path_index) && exists_test(quantizer_state_path(path_index))) {
        cout << "Loading Vamana ExRaBitQ4 index from " << path_index << ":\n";
        if (query_mode == QueryMode::ResidualRerank &&
            !exists_test(residual_state_path(path_index, query_mode))) {
            cout << "Missing Vamana residual sidecar; rebuilding staged-search index\n";
        } else {
            try {
                index.loadIndex(path_index, vecsize);
                need_build = false;
                ifstream metrics(build_metrics_path);
                if (metrics.is_open()) {
                    metrics >> build_time_ms;
                }
            } catch (const std::exception &error) {
                cout << "Existing Vamana index is incompatible: " << error.what() << "\n";
                cout << "Rebuilding Vamana ExRaBitQ4 index\n";
            }
        }
    }

    if (need_build) {
        StopW total_build_timer;
        cout << "Building Vamana ExRaBitQ4 index:\n";
        cout << "Training 1 ExRaBitQ centroid from "
             << min(centroid_train_samples, vecsize) << " base vectors\n";
        StopW train_center_timer;
        vector<float> center = train_global_center(
            input,
            vecdim,
            vecsize,
            centroid_train_samples,
            random_seed);
        index.space().setGlobalCenter(center.data());
        const string centroid_path = path_index_string + ".centroids";
        ofstream centroid_output(centroid_path, ios::binary | ios::trunc);
        if (!centroid_output.is_open()) {
            throw runtime_error("cannot create centroid file: " + centroid_path);
        }
        index.space().saveCentroids(
            centroid_output,
            static_cast<uint32_t>(random_seed),
            static_cast<uint64_t>(centroid_train_samples));
        cout << "build_stage=train_center"
             << " seconds=" << train_center_timer.getElapsedTimeMicro() / 1000000.0
             << " seconds"
             << "\n";

        FvecMmap base_vectors(path_data, vecsize, vecdim);
        const size_t full_record_size = index.space().get_full_data_size();
        const size_t compact_record_size = index.space().get_data_size();
        const string payload_path = path_index_string + ".payload.tmp";
        std::unique_ptr<DiskPayloadStore> payload_store;
        if (query_mode == QueryMode::ResidualRerank) {
            payload_store.reset(new DiskPayloadStore(payload_path, vecsize * full_record_size));
        }
        vector<char> compact_payloads(vecsize * compact_record_size, 0);
        vector<size_t> centroid_counts(1, 0);
        cout << "quantized_vamana_build_setup"
             << " graph=Vamana"
             << " payload=ExRaBitQ4"
             << " build_distance=ExRaBitQ4_symmetric"
             << " query_distance=Float32_to_ExRaBitQ4"
             << " R=" << hnswlib::VamanaIndex::kDefaultR
             << " L_build=" << hnswlib::VamanaIndex::kDefaultLBuild
             << " alpha=" << hnswlib::VamanaIndex::kDefaultAlpha
             << " beam_width=" << hnswlib::VamanaIndex::kDefaultBeamWidth
             << " full_record_bytes=" << full_record_size
             << " compact_record_bytes=" << compact_record_size
             << " residual_rerank=" << (query_mode == QueryMode::ResidualRerank ? 1 : 0)
             << " rerank_candidates=" << rerank_candidates
             << "\n";

        StopW encode_timer;
        double encode_cpu_us = 0.0;
#pragma omp parallel for
        for (int64_t label = 0; label < static_cast<int64_t>(vecsize); ++label) {
            const float *raw = base_vectors.vector(static_cast<size_t>(label));
            StopW local_timer;
            vector<char> full(full_record_size, 0);
            vector<char> compact(compact_record_size, 0);
            const uint8_t centroid_id = index.space().assignCentroid(raw);
            index.space().encodeVectorFullWithCentroid(raw, centroid_id, full.data());
            index.space().copyCompactPayloadFromFull(full.data(), compact.data());
            if (payload_store) {
                payload_store->writeRecord(
                    static_cast<size_t>(label), full.data(), full_record_size);
            }
            std::memcpy(
                compact_payloads.data() + static_cast<size_t>(label) * compact_record_size,
                compact.data(),
                compact_record_size);
#pragma omp atomic
            encode_cpu_us += local_timer.getElapsedTimeMicro();
#pragma omp atomic
            centroid_counts[centroid_id]++;
        }
        const double encode_wall_us = encode_timer.getElapsedTimeMicro();
        cout << "build_stage=payload_encode"
             << " cpu_seconds=" << encode_cpu_us / 1000000.0
             << " wall_seconds=" << encode_wall_us / 1000000.0
             << " seconds"
             << " count=" << vecsize
             << " payload=ExRaBitQ4"
             << " compact_record_bytes=" << compact_record_size
             << "\n";

        const size_t build_batch_size =
            getenv_size_t("RABITQ_VAMANA_BUILD_BATCH", 4096);
        StopW graph_timer;
        cout << "Entering Vamana segmented parallel graph build"
             << " count=" << vecsize
             << " build_batch_size=" << build_batch_size
             << " backedge_mode=grouped_parallel_batch_prune"
             << " graph=Vamana"
             << " payload=ExRaBitQ4"
             << " build_distance=ExRaBitQ4_symmetric"
             << " R=" << hnswlib::VamanaIndex::kDefaultR
             << " L_build=" << hnswlib::VamanaIndex::kDefaultLBuild
             << " alpha=" << hnswlib::VamanaIndex::kDefaultAlpha
             << " beam_width=" << hnswlib::VamanaIndex::kDefaultBeamWidth
             << "\n";
        cout.flush();
        const size_t report_every =
            getenv_size_t("RABITQ_BUILD_REPORT_EVERY", 65536);
        size_t next_report = report_every;
        try {
            index.buildEncodedSymmetricBulk(
                std::move(compact_payloads),
                vecsize,
                build_batch_size,
                [&](size_t built, size_t total) {
                    if (report_every != 0U &&
                        (built >= next_report || built == total)) {
                        cout << "Vamana graph build progress"
                             << " count=" << built
                             << " total=" << total
                             << " percent=" << (100.0 * static_cast<double>(built) /
                                                static_cast<double>(total))
                             << " kips=" << kips_from_count_us(
                                 built, graph_timer.getElapsedTimeMicro())
                             << " Mem=" << getCurrentRSS() / 1000000 << "MB\n";
                        cout.flush();
                        while (next_report <= built) {
                            next_report += report_every;
                        }
                    }
                });
        } catch (const std::exception &error) {
            cout << "build_stage=vamana_graph_build_failed"
                 << " error=" << error.what()
                 << " count=" << vecsize
                 << " build_batch_size=" << build_batch_size
                 << "\n";
            cout.flush();
            throw;
        }
        const double graph_build_ms = graph_timer.getElapsedTimeMicro() / 1000.0;
        cout << "build_stage=vamana_graph_build"
             << " seconds=" << graph_build_ms / 1000.0
             << " seconds"
             << " kips=" << kips_from_count_us(vecsize, graph_build_ms * 1000.0)
             << " build_mode=segmented_parallel_grouped_backedge"
             << " build_batch_size=" << build_batch_size
             << "\n";

        if (refine_passes != 0) {
            StopW refine_timer;
            cout << "Entering Vamana symmetric refine"
                 << " count=" << vecsize
                 << " refine_passes=" << refine_passes
                 << " build_batch_size=" << build_batch_size
                 << " backedge_mode=grouped_parallel_batch_prune"
                 << " graph=Vamana"
                 << " payload=ExRaBitQ4"
                 << " refine_distance=ExRaBitQ4_symmetric"
                 << " R=" << hnswlib::VamanaIndex::kDefaultR
                 << " L_build=" << hnswlib::VamanaIndex::kDefaultLBuild
                 << " alpha=" << hnswlib::VamanaIndex::kDefaultAlpha
                 << " beam_width=" << hnswlib::VamanaIndex::kDefaultBeamWidth
                 << "\n";
            cout.flush();
            size_t next_refine_report = report_every;
            index.refineGraphSymmetric(
                refine_passes,
                build_batch_size,
                [&](size_t done, size_t total) {
                    if (report_every != 0U &&
                        (done >= next_refine_report || done == total)) {
                        cout << "Vamana graph refine progress"
                             << " count=" << done
                             << " total=" << total
                             << " percent=" << (100.0 * static_cast<double>(done) /
                                                static_cast<double>(total))
                             << " kips=" << kips_from_count_us(
                                 done, refine_timer.getElapsedTimeMicro())
                             << " Mem=" << getCurrentRSS() / 1000000 << "MB\n";
                        cout.flush();
                        while (next_refine_report <= done) {
                            next_refine_report += report_every;
                        }
                    }
                });
            const double refine_ms = refine_timer.getElapsedTimeMicro() / 1000.0;
            cout << "build_stage=vamana_graph_refine"
                 << " seconds=" << refine_ms / 1000.0
                 << " seconds"
                 << " kips=" << kips_from_count_us(
                     vecsize * refine_passes, refine_ms * 1000.0)
                 << " refine_passes=" << refine_passes
                 << " build_mode=symmetric_full_graph_refine_grouped_backedge"
                 << " build_batch_size=" << build_batch_size
                 << "\n";
        }

        if (query_mode == QueryMode::ResidualRerank) {
            StopW residual_timer;
            const string residual_path = residual_state_path(path_index, query_mode);
            index.materializeExternalResidualsFromFullPayloadFile(
                payload_path,
                residual_path,
                full_record_size);
            const double residual_ms = residual_timer.getElapsedTimeMicro() / 1000.0;
            cout << "build_stage=materialize_residual_sidecar"
                 << " seconds=" << residual_ms / 1000.0
                 << " seconds"
                 << " residual_path=" << residual_path
                 << " residual_bytes=" << file_size_bytes(residual_path)
                 << "\n";
            std::remove(payload_path.c_str());
        }

        StopW save_timer;
        index.saveIndex(path_index);
        const double save_ms = save_timer.getElapsedTimeMicro() / 1000.0;
        build_time_ms = total_build_timer.getElapsedTimeMicro() / 1000.0;
        ofstream metrics(build_metrics_path, ios::trunc);
        if (metrics.is_open()) {
            metrics << build_time_ms << '\n';
        }
        cout << "build_stage=save_index"
             << " seconds=" << save_ms / 1000.0
             << " seconds\n";
        cout << "build_time_seconds=" << build_time_ms / 1000.0
             << " seconds\n";
        print_vamana_index_storage(path_index_string, index);
    }

    if (build_time_ms < 0.0) {
        build_time_ms = 0.0;
    }

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    const size_t k = 10;
    cout << "Parsing gt:\n";
    get_gt(massQA, qsize, gt_width, answers, k);
    cout << "Loaded gt\n";
    test_vs_recall_vamana(
        massQ,
        qsize,
        index,
        answers,
        k,
        path_index_string,
        build_time_ms,
        rerank_candidates,
        query_mode,
        refine_passes,
        paper_prune_active,
        paper_epsilon0);

    delete[] massQA;
    delete[] massQ;
}

static void report_paper_true_distance_coverage(
    float *massQ,
    size_t qsize,
    size_t vecdim,
    const string &base_path,
    const vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    RaBitQHierarchicalNSW &index) {
    std::ifstream base(base_path, std::ios::binary);
    if (!base.is_open()) throw std::runtime_error("cannot open base file for paper coverage test");
    const float epsilons[] = {1.9f, 2.2f, 2.5f};
    size_t checked[3] = {0, 0, 0};
    size_t violations[3] = {0, 0, 0};
    std::vector<float> point(vecdim);
    const std::streamoff record_bytes = static_cast<std::streamoff>(sizeof(uint32_t) +
        vecdim * sizeof(float));
    for (size_t qi = 0; qi < qsize; ++qi) {
        const void *prepared = index.space().prepare_query(massQ + qi * vecdim);
        auto gt = answers[qi];
        while (!gt.empty()) {
            const labeltype label = gt.top().second;
            gt.pop();
            base.clear();
            base.seekg(static_cast<std::streamoff>(label) * record_bytes, std::ios::beg);
            uint32_t stored_dim = 0;
            base.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
            if (!base.good() || stored_dim != vecdim)
                throw std::runtime_error("base file error during paper coverage test");
            base.read(reinterpret_cast<char *>(point.data()),
                      static_cast<std::streamsize>(vecdim * sizeof(float)));
            if (!base.good()) throw std::runtime_error("base vector read failed");
            double true_distance = 0.0;
            for (size_t d = 0; d < vecdim; ++d) {
                const double delta = static_cast<double>(point[d]) - massQ[qi * vecdim + d];
                true_distance += delta * delta;
            }
            const std::vector<char> encoded = index.space().encodeVector(point.data());
            for (size_t ei = 0; ei < 3; ++ei) {
                const auto paper = index.space().compute_paper_prune_estimate(
                    prepared, encoded.data(), epsilons[ei]);
                if (!paper.valid) continue;
                ++checked[ei];
                const double tolerance = 1e-5 * std::max(1.0, std::fabs(true_distance));
                if (static_cast<double>(paper.lower_bound) > true_distance + tolerance)
                    ++violations[ei];
            }
        }
        index.space().release_query(prepared);
    }
    for (size_t ei = 0; ei < 3; ++ei) {
        cout << "paper_offline_true_distance_coverage"
             << " epsilon0=" << epsilons[ei]
             << " checked=" << checked[ei]
             << " paper_bound_violation_true_distance=" << violations[ei]
             << " violation_ratio="
             << (checked[ei] ? static_cast<double>(violations[ei]) / checked[ei] : 0.0)
             << " sample=ground_truth_pairs"
             << " included_in_search_timing=0\n";
    }
}

void sift_test1B() {
    sift_vamana_test();
    return;

    const string abc_ablation = getenv_string("RABITQ_ABC_ABLATION", "");
    if (!abc_ablation.empty() && abc_ablation != "A" &&
        abc_ablation != "B" && abc_ablation != "C" && abc_ablation != "D")
        throw runtime_error("RABITQ_ABC_ABLATION must be A, B, C, or D");
    const bool paper_prune_profile = []() {
        const char *value = std::getenv("RABITQ_PAPER_PRUNE_COMPARE");
        return value != nullptr && value[0] != '\0';
    }();
    const int efConstruction = static_cast<int>(getenv_size_t(
        "RABITQ_EF_CONSTRUCTION", paper_prune_profile ? 200 : 400));
    const int M = static_cast<int>(getenv_size_t(
        "RABITQ_M", paper_prune_profile ? 16 : 32));
    const int centroid_count = static_cast<int>(getenv_size_t(
        "RABITQ_CENTROID_COUNT", paper_prune_profile ? 1 : 256));
    const size_t rerank_candidates =
        getenv_size_t("RABITQ_RERANK_CANDIDATES", 100);
    const int random_seed = 100;
    const string code_layout_name = getenv_string("RABITQ_CODE_LAYOUT", "sequential");
    const RaBitQCodeLayout code_layout = code_layout_name == "sequential"
        ? RaBitQCodeLayout::SequentialNibble
        : (code_layout_name == "turbo128"
            ? RaBitQCodeLayout::Turbo128
            : throw runtime_error("RABITQ_CODE_LAYOUT must be sequential or turbo128"));

    struct DatasetConfig {
        string name;
        size_t dim;
        size_t gt_width;
        string base_path;
        string query_path;
        string gt_path;
        string index_prefix;
    };

    const string dbpedia_data_dir = "/home/kai3/coco/data/dbpedia_openai1536";
    const string dataset_name_config = getenv_string(
        "RABITQ_DATASET", "dbpedia_openai1536");
    const bool is_dbpedia_dataset =
        dataset_name_config == "dbpedia_openai1536" ||
        dataset_name_config == "dbpedia" ||
        dataset_name_config == "dbpedia-openai1536";
    const DatasetConfig dataset{
        dataset_name_config,
        getenv_size_t("RABITQ_DIM", is_dbpedia_dataset ? 1536 : 128),
        getenv_size_t("RABITQ_GT_WIDTH", 100),
        getenv_string(
            "RABITQ_BASE_PATH",
            is_dbpedia_dataset
                ? dbpedia_data_dir + "/dbpedia_openai1536_base.fvecs"
                : "/home/kai3/coco/data/sift10m/sift10m_base.fvecs"),
        getenv_string(
            "RABITQ_QUERY_PATH",
            is_dbpedia_dataset
                ? dbpedia_data_dir + "/dbpedia_openai1536_query.fvecs"
                : "/home/kai3/coco/data/sift10m/sift10m_query.fvecs"),
        getenv_string(
            "RABITQ_GT_PATH",
            is_dbpedia_dataset
                ? dbpedia_data_dir + "/dbpedia_openai1536_groundtruth.ivecs"
                : "/home/kai3/coco/data/sift10m/sift10m_groundtruth.ivecs"),
        dataset_name_config};

    const char *dataset_name = dataset.name.c_str();
    const size_t vecdim = dataset.dim;
    const size_t gt_width = dataset.gt_width;
    const bool external_residual_storage = true;
    const QueryMode query_mode = configured_query_mode();
    const size_t residual_bits = configured_residual_bits(4);
    if (residual_bits != 4 && residual_bits != 8) {
        throw runtime_error("Only primary4+residual4 and primary4+residual8 are supported");
    }
    const size_t default_centroid_train_samples = dataset.name == "sift10m"
        ? 10000000
        : std::min<size_t>(100000, fvec_count_from_file_size(dataset.base_path.c_str(), vecdim));
    const size_t centroid_train_samples =
        getenv_size_t("RABITQ_CENTROID_TRAIN_SAMPLES", default_centroid_train_samples);
    const hnswlib::RaBitQSpace::ResidualQuantizationConfig residual_config =
        [&]() {
            hnswlib::RaBitQSpace::ResidualQuantizationConfig config;
            config.bits = residual_bits == 4
                ? hnswlib::RaBitQSpace::ResidualQuantizationBits::B4
                : hnswlib::RaBitQSpace::ResidualQuantizationBits::B8;
            const size_t abc_default_block = abc_ablation == "B" ? 2048 : 16;
            config.block_size = getenv_size_t(
                "RABITQ_RESIDUAL_BLOCK_SIZE", abc_default_block);
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
            config.mse_optimal_scale = residual_bits == 4;
            config.scale_fp16 = residual_bits == 4;
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

    const string build_distance_mode = getenv_string(
        "RABITQ_BUILD_DISTANCE",
        centroid_count == 1
            ? "asymmetric4"
            : "float32");
    if (build_distance_mode != "asymmetric4" &&
        build_distance_mode != "symmetric4" && build_distance_mode != "float32")
        throw runtime_error(
            "RABITQ_BUILD_DISTANCE must be asymmetric4, symmetric4, or float32");
    const bool asymmetric4_build = build_distance_mode == "asymmetric4";
    const bool symmetric4_build = build_distance_mode == "symmetric4";
    const bool quantized4_build = asymmetric4_build || symmetric4_build;
    const bool force_rebuild = getenv_bool01_strict("RABITQ_FORCE_REBUILD", false);
    if (quantized4_build && centroid_count != 1)
        throw runtime_error("4-bit construction requires K=1");
    if (symmetric4_build && code_layout != RaBitQCodeLayout::SequentialNibble)
        throw runtime_error("symmetric 4-bit construction requires Sequential layout");
    if (!abc_ablation.empty()) {
        if (abc_ablation == "D" && !symmetric4_build)
            throw runtime_error("D requires RABITQ_BUILD_DISTANCE=symmetric4");
        if (abc_ablation != "D" && !asymmetric4_build)
            throw runtime_error("A, B, and C require RABITQ_BUILD_DISTANCE=asymmetric4");
    }

    char index_name[1024];
    const char *path_q = dataset.query_path.c_str();
    const char *path_data = dataset.base_path.c_str();
    const char *path_gt = dataset.gt_path.c_str();
    const size_t vecsize = fvec_count_from_file_size(path_data, vecdim);
    const size_t qsize = fvec_count_from_file_size(path_q, vecdim);
    const char *scale_name = residual_config.mse_optimal_scale ? "mse" : "maxabs";
    const char *scale_storage_name = residual_config.scale_fp16 ? "fp16" : "fp32";
    const string block_suffix = residual_config.block_size == 16
        ? ""
        : "_block" + std::to_string(residual_config.block_size);
    snprintf(
        index_name,
        sizeof(index_name),
        "%s_primary4_residual%zu_trueK%d_%s_%s_%s%s_%s_ef_%d_M_%d.bin",
        dataset.index_prefix.c_str(),
        residual_bits,
        centroid_count,
        scale_name,
        scale_storage_name,
        code_layout_name.c_str(),
        block_suffix.c_str(),
        asymmetric4_build ? "asym4bitbuild" :
            (symmetric4_build ? "sym4bitbuild" : "floatbuild"),
        efConstruction,
        M);
    const string default_index_dir = paper_prune_profile
        ? "build/indexes/dbpedia_ablation"
        : (residual_bits == 8
            ? "build/trueK256_train10m_residual8"
            : "build/trueK256_train10m");
    const string index_dir = getenv_string("RABITQ_INDEX_DIR", default_index_dir);
    ensure_directory_exists(index_dir);
    const string path_index_string = join_path(index_dir, index_name);
    const char *path_index = path_index_string.c_str();

    if (paper_prune_profile) {
        cout << "experiment_profile=paper_prune_dbpedia_k1"
             << " defaults=dim1536,gt10,K1,M16,efConstruction200,sequential,residual4,mse,fp16,rerank100"
             << " explicit_environment_overrides=enabled"
             << " force_rebuild=" << (force_rebuild ? 1 : 0) << "\n";
    }
    cout << "build_runtime"
         << " omp_max_threads=" << omp_get_max_threads()
         << " omp_dynamic=" << omp_get_dynamic()
         << "\n";

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
        external_residual_storage,
        asymmetric4_build ? "extended_rabitq_asymmetric4" :
            (symmetric4_build ? "extended_rabitq_symmetric4" : "float32_l2"));

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
    appr_alg->space().set_code_layout(code_layout);
    const string centroid_query_mode = getenv_string("RABITQ_CENTROID_MODE", "eager");
    if (centroid_query_mode == "lazy") {
        appr_alg->space().set_centroid_query_mode(RaBitQSpace::CentroidQueryMode::Lazy);
    } else if (centroid_query_mode != "eager") {
        throw runtime_error("RABITQ_CENTROID_MODE must be eager or lazy");
    }
    cout << "  encoded_bytes_per_vector=" << appr_alg->space().get_data_size()
         << (external_residual_storage
                ? " (4-bit code in index; residual in mmap sidecar)\n"
                : " (4-bit code + residual code)\n");
    cout << "  residual_bits=" << appr_alg->space().get_residual_bits()
         << " residual_block_size=" << appr_alg->space().get_residual_block_size()
         << " residual_scale_mode="
         << (appr_alg->space().get_residual_mse_optimal_scale() ? "mse" : "max_abs")
         << " primary_bits=" << hnswlib::RaBitQSpace::kTotalBits
         << " short_bits=" << hnswlib::RaBitQSpace::kShortBits
         << " remaining_bits=" << hnswlib::RaBitQSpace::kRemainingBits
         << " compact_primary_bytes_per_vector=" << appr_alg->space().get_compact_code_bytes()
         << " residual_code_bytes_per_vector=" << appr_alg->space().get_residual_code_bytes()
         << " residual_record_bytes_per_vector=" << appr_alg->space().get_residual_disk_record_bytes()
         << " primary_4bit_empirical_error_reference="
         << appr_alg->space().get_primary_code_empirical_error_reference()
         << "\n";

    const string build_metrics_path = string(path_index) + ".build_metrics";
    double reported_graph_construction_us = -1.0;
    double reported_total_build_us = -1.0;
    const auto load_build_metrics = [](const string &path, double &graph_us, double &total_us) {
        ifstream metrics(path);
        if (metrics.is_open()) metrics >> graph_us >> total_us;
    };
    const auto save_build_metrics = [](const string &path, double graph_us, double total_us) {
        ofstream metrics(path, ios::trunc);
        if (!metrics.is_open()) throw runtime_error("cannot write build metrics: " + path);
        metrics << graph_us << ' ' << total_us << '\n';
    };

    bool need_build = true;
    if (!force_rebuild && exists_test(path_index)) {
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
                appr_alg->space().set_code_layout(code_layout);
                cout << "  encoded_bytes_per_vector=" << appr_alg->space().get_data_size()
                     << (external_residual_storage
                            ? " (4-bit code in index; residual in mmap sidecar)\n"
                            : " (4-bit code + residual code)\n");
                cout << "  residual_bits=" << appr_alg->space().get_residual_bits()
                     << " residual_block_size=" << appr_alg->space().get_residual_block_size()
                     << " residual_scale_mode="
                     << (appr_alg->space().get_residual_mse_optimal_scale() ? "mse" : "max_abs")
                     << " primary_bits=" << hnswlib::RaBitQSpace::kTotalBits
                     << " short_bits=" << hnswlib::RaBitQSpace::kShortBits
                     << " remaining_bits=" << hnswlib::RaBitQSpace::kRemainingBits
                     << " compact_primary_bytes_per_vector=" << appr_alg->space().get_compact_code_bytes()
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
        const string centroid_path = getenv_string(
            "RABITQ_CENTROID_PATH", path_index_string + ".centroids");
        if (!force_rebuild && exists_test(centroid_path)) {
            ifstream centroid_input(centroid_path, ios::binary);
            uint32_t stored_seed = 0;
            uint64_t stored_samples = 0;
            appr_alg->space().loadCentroids(
                centroid_input, &stored_seed, &stored_samples);
            if (stored_seed != static_cast<uint32_t>(random_seed) ||
                stored_samples != centroid_train_samples) {
                throw runtime_error("centroid training metadata does not match requested configuration");
            }
            cout << "Loaded centroids from " << centroid_path << "\n";
        } else {
            if (centroid_count == 1) {
                centroids = train_global_center(
                    input,
                    vecdim,
                    vecsize,
                    centroid_train_samples,
                    random_seed);
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
            ofstream centroid_output(centroid_path, ios::binary | ios::trunc);
            if (!centroid_output.is_open()) {
                throw runtime_error("cannot create centroid file: " + centroid_path);
            }
            appr_alg->space().saveCentroids(
                centroid_output,
                static_cast<uint32_t>(random_seed),
                static_cast<uint64_t>(centroid_train_samples));
            cout << "Saved centroids to " << centroid_path << "\n";
        }
        input.clear();
        input.seekg(0, ios::beg);
        const double train_center_us = train_center_timer.getElapsedTimeMicro();
        cout << "build_stage=train_center"
             << " us=" << train_center_us << "\n";
        vector<size_t> centroid_counts(static_cast<size_t>(centroid_count), 0);

        if (quantized4_build) {
            cout << "build_distance="
                 << (asymmetric4_build
                     ? "extended_rabitq_asymmetric4"
                     : "extended_rabitq_symmetric4")
                 << " build_query=" << (asymmetric4_build ? "float32" : "primary4")
                 << " build_database=primary4"
                 << " build_residual_used=0"
                 << " build_float32_retained=0"
                 << " symmetric_build_prepared=" << (symmetric4_build ? 1 : 0)
                 << "\n";
            const size_t full_record_size = appr_alg->space().get_full_data_size();
            const size_t compact_record_size = appr_alg->space().get_data_size();
            const string payload_path = string(path_index) + ".payload.tmp";
            const string shared_graph_path = join_path(
                index_dir,
                dataset.name + "_primary4_K1_" + code_layout_name +
                    (asymmetric4_build ? "_asym4bit_graph_seed_" : "_sym4bit_graph_seed_") +
                    std::to_string(random_seed) +
                    "_train_" + std::to_string(centroid_train_samples) +
                    "_ef_" + std::to_string(efConstruction) +
                    "_M_" + std::to_string(M) + ".bin");
            const bool reuse_quantized_graph =
                !force_rebuild && exists_test(shared_graph_path);
            const string shared_graph_metrics_path = shared_graph_path + ".build_metrics";
            DiskPayloadStore payload_store(payload_path, vecsize * full_record_size);
            FvecMmap base_vectors(path_data, vecsize, vecdim);
            cout << "quantized_graph_build_setup"
                 << " payload_path=" << payload_path
                 << " full_record_bytes=" << full_record_size
                 << " compact_record_bytes=" << compact_record_size
                 << " payload_bytes=" << vecsize * full_record_size
                 << " compact_payload_bytes="
                 << (reuse_quantized_graph ? 0 : vecsize * compact_record_size)
                 << " shared_graph_path=" << shared_graph_path
                 << " reuse_graph=" << (reuse_quantized_graph ? 1 : 0)
                 << "\n";
            if (reuse_quantized_graph) {
                cout << "Loading shared " << build_distance_mode
                     << " graph from " << shared_graph_path << "\n";
            } else if (asymmetric4_build) {
                appr_alg->setAsymmetricBuildRawProvider(
                    [&base_vectors](labeltype label) -> const void * {
                        return base_vectors.vector(static_cast<size_t>(label));
                    });
            } else if (symmetric4_build) {
                appr_alg->setSymmetricBuildPrepared(true);
            }
            StopW encode_timer;
            double encode_cpu_us = 0.0;
            double payload_write_cpu_us = 0.0;
            vector<char> compact_payloads;
            if (!reuse_quantized_graph) {
                compact_payloads.assign(vecsize * compact_record_size, 0);
            }
            cout << "Entering payload encode loop"
                 << " count=" << vecsize
                 << "\n";
#pragma omp parallel for
            for (int64_t label = 0; label < static_cast<int64_t>(vecsize); ++label) {
                const float *raw = base_vectors.vector(static_cast<size_t>(label));
                StopW local_encode_timer;
                vector<char> full(full_record_size, 0);
                vector<char> compact(compact_record_size, 0);
                const uint8_t centroid_id = appr_alg->space().assignCentroid(raw);
                appr_alg->space().encodeVectorFullWithCentroid(raw, centroid_id, full.data());
                appr_alg->space().copyCompactPayloadFromFull(full.data(), compact.data());
                const double local_encode_us = local_encode_timer.getElapsedTimeMicro();
                StopW local_write_timer;
                payload_store.writeRecord(static_cast<size_t>(label), full.data(), full_record_size);
                const double local_write_us = local_write_timer.getElapsedTimeMicro();
                if (!reuse_quantized_graph) {
                    std::memcpy(
                        compact_payloads.data() + static_cast<size_t>(label) * compact_record_size,
                        compact.data(),
                        compact_record_size);
                }
#pragma omp atomic
                encode_cpu_us += local_encode_us;
#pragma omp atomic
                payload_write_cpu_us += local_write_us;
#pragma omp atomic
                centroid_counts[centroid_id]++;
            }
            const double encode_wall_us = encode_timer.getElapsedTimeMicro();
            cout << "build_stage=payload_encode"
                 << " cpu_us=" << encode_cpu_us
                 << " count=" << vecsize
                 << " full_record_bytes=" << full_record_size
                 << " compact_record_bytes=" << compact_record_size << "\n";
            cout << "build_stage=payload_write"
                 << " cpu_us=" << payload_write_cpu_us
                 << " payload_bytes=" << vecsize * full_record_size
                 << "\n";
            cout << "build_stage=payload_encode_and_write_wall"
                 << " wall_us=" << encode_wall_us
                 << "\n";

            double graph_build_us = 0.0;
            if (!reuse_quantized_graph) {
                StopW graph_timer;
                const size_t report_every = getenv_size_t("RABITQ_BUILD_REPORT_EVERY", 0);
                cout << "Entering " << (asymmetric4_build ? "asymmetric" : "symmetric")
                     << " graph addpoint loop"
                     << " count=" << vecsize
                     << " report_every=" << report_every
                     << "\n";
#pragma omp parallel for
                for (int64_t label = 0; label < static_cast<int64_t>(vecsize); ++label) {
                    const float *raw = base_vectors.vector(static_cast<size_t>(label));
                    const char *compact =
                        compact_payloads.data() + static_cast<size_t>(label) * compact_record_size;
                    if (asymmetric4_build) {
                        appr_alg->addPointAsymmetric(
                            raw, static_cast<labeltype>(label), compact);
                    } else {
                        appr_alg->index().addPoint(
                            compact, static_cast<labeltype>(label));
                    }
                    if (report_every != 0U &&
                        (static_cast<size_t>(label) + 1U) % report_every == 0U) {
#pragma omp critical
                        cout << (asymmetric4_build ? "Asymmetric" : "Symmetric")
                             << " graph addpoint progress label=" << (label + 1)
                             << " count=" << appr_alg->index().cur_element_count << "\n";
                    }
                }
                graph_build_us = graph_timer.getElapsedTimeMicro();
                vector<char>().swap(compact_payloads);
            }
            StopW residual_timer;
            if (reuse_quantized_graph) {
                double ignored_total_us = -1.0;
                load_build_metrics(
                    shared_graph_metrics_path,
                    reported_graph_construction_us,
                    ignored_total_us);
                HierarchicalNSW<float> shared_graph(
                    &appr_alg->space(), shared_graph_path, false, vecsize);
                appr_alg->importGraphFromFloatIndexWithFullPayloadFileAndExternalResiduals(
                    shared_graph,
                    payload_path,
                    residual_state_path(path_index, query_mode),
                    full_record_size,
                    true);
                cout << "build_stage=" << build_distance_mode << "_graph_reuse"
                     << " us=" << graph_build_us
                     << " source=" << shared_graph_path << "\n";
            } else {
                reported_graph_construction_us = graph_build_us;
                cout << "build_stage=" << build_distance_mode << "_graph_build"
                     << " us=" << graph_build_us
                     << " kips=" << kips_from_count_us(vecsize, graph_build_us)
                     << " asymmetric_distance_calls=" << appr_alg->asymmetricBuildDistanceCalls()
                     << " encoded_distance_calls=" << appr_alg->encodedBuildDistanceCalls()
                     << " symmetric_prepared_distance_calls="
                     << appr_alg->symmetricPreparedBuildDistanceCalls() << "\n";
                if (asymmetric4_build && appr_alg->encodedBuildDistanceCalls() != 0)
                    throw runtime_error("asymmetric construction used encoded-to-encoded distance");
                if (symmetric4_build &&
                    (appr_alg->asymmetricBuildDistanceCalls() != 0 ||
                     appr_alg->encodedBuildDistanceCalls() == 0 ||
                     appr_alg->symmetricPreparedBuildDistanceCalls() == 0))
                    throw runtime_error("symmetric construction did not exclusively use encoded distance");
                appr_alg->index().saveIndex(shared_graph_path);
                save_build_metrics(
                    shared_graph_metrics_path,
                    reported_graph_construction_us,
                    reported_graph_construction_us);
                cout << "Saved shared " << build_distance_mode
                     << " graph to " << shared_graph_path << "\n";
            }
            if (!reuse_quantized_graph) {
                appr_alg->materializeExternalResidualsFromFullPayloadFile(
                    payload_path,
                    residual_state_path(path_index, query_mode),
                    full_record_size);
            }
            cout << "build_stage=residual_sidecar_materialize"
                 << " us=" << residual_timer.getElapsedTimeMicro()
                 << " count=" << vecsize << "\n";
            if (asymmetric4_build && !reuse_quantized_graph)
                appr_alg->clearAsymmetricBuildRawProvider();
            if (symmetric4_build && !reuse_quantized_graph)
                appr_alg->setSymmetricBuildPrepared(false);
            std::remove(payload_path.c_str());
            StopW save_index_timer;
            appr_alg->saveIndex(path_index);
            cout << "build_stage=save_index us=" << save_index_timer.getElapsedTimeMicro() << "\n";
            reported_total_build_us = total_build_timer.getElapsedTimeMicro();
            save_build_metrics(
                build_metrics_path,
                reported_graph_construction_us,
                reported_total_build_us);
            cout << "build_total_us=" << reported_total_build_us << "\n";
            print_index_file_size(path_index, query_mode);
        } else {

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

        cout << "Building HNSW graph with float32 L2 distances, then encoding payloads to 4-bit RaBitQ\n";
        L2Space float_space(vecdim);
        const string float_graph_path = getenv_string(
            "RABITQ_FLOAT_GRAPH_PATH",
            join_path(index_dir, dataset.name + "_float_ef_" +
                std::to_string(efConstruction) + "_M_" + std::to_string(M) + ".bin"));
        const string float_graph_metrics_path = float_graph_path + ".build_metrics";
        const bool reuse_float_graph = exists_test(float_graph_path);
        unique_ptr<HierarchicalNSW<float>> float_index;
        if (reuse_float_graph) {
            cout << "Loading shared Float32 graph from " << float_graph_path << "\n";
            float_index.reset(new HierarchicalNSW<float>(
                &float_space, float_graph_path, false, vecsize));
            if (float_index->cur_element_count != vecsize) {
                throw runtime_error("shared Float32 graph element count mismatch");
            }
        } else {
            float_index.reset(new HierarchicalNSW<float>(
                &float_space, vecsize, M, efConstruction, random_seed));
        }

        cout << "Loading base vectors before graph timing:\n";
        vector<float> massB = load_fvecs_raw(path_data, vecsize, vecdim);
        input.close();
        cout << "Loaded base vectors"
             << " count=" << vecsize
             << " bytes=" << massB.size() * sizeof(float)
             << " Mem: " << getCurrentRSS() / 1000000 << " Mb\n";

        double float_graph_build_us = -1.0;
        if (reuse_float_graph) {
            double ignored_total_us = -1.0;
            load_build_metrics(float_graph_metrics_path, float_graph_build_us, ignored_total_us);
            reported_graph_construction_us = float_graph_build_us;
            cout << "build_stage=float_graph_reuse"
                 << " us=0"
                 << " source=" << float_graph_path << "\n";
        } else {
            StopW float_graph_timer;
            float_index->addPoint(massB.data(), (size_t)0);
#pragma omp parallel for
            for (int i = 1; i < static_cast<int>(vecsize); i++) {
                float_index->addPoint(
                    massB.data() + static_cast<size_t>(i) * vecdim,
                    (size_t)i);
                if ((static_cast<size_t>(i) + 1U) % report_every == 0U) {
#pragma omp critical
                    cout << (static_cast<size_t>(i) + 1U) / (0.01 * vecsize) << " %, "
                         << kips_from_count_us(
                                static_cast<size_t>(i) + 1U,
                                float_graph_timer.getElapsedTimeMicro())
                         << " kips "
                         << " Mem: " << getCurrentRSS() / 1000000 << " Mb \n";
                }
            }
            float_graph_build_us = float_graph_timer.getElapsedTimeMicro();
            reported_graph_construction_us = float_graph_build_us;
            float_index->saveIndex(float_graph_path);
            save_build_metrics(float_graph_metrics_path, float_graph_build_us, float_graph_build_us);
            cout << "Saved shared Float32 graph to " << float_graph_path << "\n";
        }

        StopW payload_encode_wall_timer;
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(vecsize); i++) {
            const size_t label = static_cast<size_t>(i);
            const float *raw = massB.data() + label * vecdim;
            StopW payload_timer;
            vector<char> encoded_payload(payload_record_size, 0);
            const uint8_t centroid_id = appr_alg->space().assignCentroid(raw);
#pragma omp atomic
            centroid_counts[centroid_id]++;
            if (external_residual_storage) {
                appr_alg->space().encodeVectorFullWithCentroid(raw, centroid_id, encoded_payload.data());
            } else {
                appr_alg->space().encodeVector(raw, encoded_payload.data());
            }
            if (payload_disk_mode) {
                disk_payload->writeRecord(label, encoded_payload.data(), payload_record_size);
            } else {
                std::memcpy(
                    payloads.data() + label * payload_record_size,
                    encoded_payload.data(),
                    payload_record_size);
            }
            const double local_payload_encode_us = payload_timer.getElapsedTimeMicro();
#pragma omp atomic
            payload_encode_cpu_us += local_payload_encode_us;
        }
        const double payload_encode_wall_us = payload_encode_wall_timer.getElapsedTimeMicro();

        cout << "Float graph build time:";
        if (float_graph_build_us >= 0.0)
            cout << 1e-6 * float_graph_build_us << " seconds";
        else
            cout << " unavailable (shared graph without build metrics)";
        cout << "; importing graph and writing 4-bit payloads\n";
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
                *float_index,
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
                *float_index,
                compact_payloads,
                appr_alg->space().get_data_size(),
                true);
            appr_alg->space().openExternalResidualStorage(residual_path, vecsize);
        } else {
            if (payload_disk_mode) {
                appr_alg->importGraphFromFloatIndexWithPayloadFile(
                    *float_index,
                    payload_path,
                    payload_record_size,
                    true);
                disk_payload.reset();
                std::remove(payload_path.c_str());
            } else {
                appr_alg->importGraphFromFloatIndexWithPayloads(
                    *float_index,
                    payloads,
                    payload_record_size,
                    true);
            }
        }
        const double graph_payload_import_us = convertw.getElapsedTimeMicro();
        const size_t graph_payload_import_count = float_index->cur_element_count;
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
        reported_total_build_us = total_build_timer.getElapsedTimeMicro();
        save_build_metrics(
            build_metrics_path,
            reported_graph_construction_us,
            reported_total_build_us);
        cout << "build_total_us=" << reported_total_build_us << "\n";
        print_index_file_size(path_index, query_mode);
        }
    }

    if (reported_graph_construction_us < 0.0 || reported_total_build_us < 0.0)
        load_build_metrics(
            build_metrics_path,
            reported_graph_construction_us,
            reported_total_build_us);
    cout << "Graph construction time: ";
    if (reported_graph_construction_us >= 0.0)
        cout << 1e-6 * reported_graph_construction_us << " seconds\n";
    else
        cout << "unavailable (legacy index without build metrics)\n";
    cout << "Build time: ";
    if (reported_total_build_us >= 0.0)
        cout << 1e-6 * reported_total_build_us << " seconds\n";
    else
        cout << "unavailable (legacy index without build metrics)\n";

    appr_alg->space().set_centroid_query_mode(
        centroid_query_mode == "lazy"
            ? RaBitQSpace::CentroidQueryMode::Lazy
            : RaBitQSpace::CentroidQueryMode::Eager);
    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    const size_t k = 10;
    cout << "Parsing gt:\n";
    get_gt(massQA, qsize, gt_width, answers, k);
    cout << "Loaded gt\n";
    const string short_shadow_compare = getenv_string("RABITQ_SHORT_SHADOW_COMPARE", "");
    const string two_bit_shadow_compare = getenv_string("RABITQ_TWO_BIT_SHADOW_COMPARE", "");
    const string paper_prune_compare = getenv_string("RABITQ_PAPER_PRUNE_COMPARE", "");
    const unsigned low_bit_experiments = static_cast<unsigned>(!short_shadow_compare.empty()) +
        static_cast<unsigned>(!two_bit_shadow_compare.empty()) +
        static_cast<unsigned>(!paper_prune_compare.empty());
    if (low_bit_experiments > 1)
        throw runtime_error("low-bit experiment variables are mutually exclusive");
    if (!paper_prune_compare.empty()) {
        if (paper_prune_compare != "baseline" && paper_prune_compare != "shadow" &&
            paper_prune_compare != "active" && paper_prune_compare != "all")
            throw runtime_error("RABITQ_PAPER_PRUNE_COMPARE must be baseline, shadow, active, or all");
        if (centroid_count != 1 || code_layout != RaBitQCodeLayout::SequentialNibble)
            throw runtime_error("paper pruning requires K=1 and Sequential layout");
        const float epsilon0 = getenv_float("RABITQ_PAPER_EPSILON0", 1.9f);
        if (!(epsilon0 == 1.9f || epsilon0 == 2.2f || epsilon0 == 2.5f))
            throw runtime_error("RABITQ_PAPER_EPSILON0 must be 1.9, 2.2, or 2.5");
        if (paper_prune_compare == "shadow" || paper_prune_compare == "all") {
            report_paper_true_distance_coverage(
                massQ, qsize, vecdim, path_data, answers, *appr_alg);
        }
        GraphTurboConfig config = appr_alg->getGraphTurboConfig();
        config.mode = GraphTurboMode::BatchPrefetch;
        config.prefetch_distance = 2;
        config.short_shadow = false;
        config.two_bit_shadow = false;
        config.paper_shadow = false;
        config.paper_active = false;
        config.paper_staged_control = false;
        config.paper_epsilon0 = epsilon0;
        const auto run = [&](const string &name, bool shadow, bool active, bool staged) {
            config.paper_shadow = shadow;
            config.paper_active = active;
            config.paper_staged_control = staged;
            appr_alg->setGraphTurboConfig(config);
            cout << "paper_experiment=" << name
                 << " epsilon0=" << epsilon0
                 << " msb_remaining_physical_layout=split_msb_sidecar_plus_full_nibble"
                 << " paper_msb_sidecar_bytes="
                 << vecsize * (appr_alg->space().paper_msb_code_bytes() +
                               sizeof(hnswlib::PaperPruneFactors<float>))
                 << " residual_sidecar_accessed_by_paper_prune=0\n";
            test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k,
                           rerank_candidates, query_mode, name);
        };
        if (paper_prune_compare == "baseline" || paper_prune_compare == "all")
            run("paper_batch_baseline", false, false, false);
        if (paper_prune_compare == "all")
            run("paper_staged_full_control", false, false, true);
        if (paper_prune_compare == "shadow" || paper_prune_compare == "all")
            run("paper_shadow", true, false, false);
        if (paper_prune_compare == "active" || paper_prune_compare == "all")
            run("paper_active", false, true, false);
        print_index_file_size(path_index, query_mode);
        return;
    }
    if (!two_bit_shadow_compare.empty()) {
        if (two_bit_shadow_compare != "baseline" && two_bit_shadow_compare != "shadow" &&
            two_bit_shadow_compare != "both")
            throw runtime_error("RABITQ_TWO_BIT_SHADOW_COMPARE must be baseline, shadow, or both");
        if (centroid_count != 1 || code_layout != RaBitQCodeLayout::SequentialNibble)
            throw runtime_error("2-bit shadow experiment requires K=1 and Sequential layout");
        GraphTurboConfig shadow_config = appr_alg->getGraphTurboConfig();
        shadow_config.mode = GraphTurboMode::BatchPrefetch;
        shadow_config.prefetch_distance = 2;
        shadow_config.short_shadow = false;
        shadow_config.two_bit_shadow = false;
        appr_alg->setGraphTurboConfig(shadow_config);
        if (two_bit_shadow_compare == "baseline" || two_bit_shadow_compare == "both") {
            test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k,
                           rerank_candidates, query_mode, "baseline");
        }
        if (two_bit_shadow_compare == "shadow" || two_bit_shadow_compare == "both") {
            shadow_config.two_bit_shadow = true;
            appr_alg->setGraphTurboConfig(shadow_config);
            test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k,
                           rerank_candidates, query_mode, "two_bit_shadow");
        }
        print_index_file_size(path_index, query_mode);
        return;
    }
    if (!short_shadow_compare.empty()) {
        if (short_shadow_compare != "baseline" && short_shadow_compare != "shadow" &&
            short_shadow_compare != "both")
            throw runtime_error("RABITQ_SHORT_SHADOW_COMPARE must be baseline, shadow, or both");
        if (centroid_count != 1 || code_layout != RaBitQCodeLayout::SequentialNibble)
            throw runtime_error("short-shadow experiment requires K=1 and Sequential layout");
        GraphTurboConfig shadow_config = appr_alg->getGraphTurboConfig();
        shadow_config.mode = GraphTurboMode::BatchPrefetch;
        shadow_config.prefetch_distance = 2;
        shadow_config.short_shadow = false;
        shadow_config.two_bit_shadow = false;
        appr_alg->setGraphTurboConfig(shadow_config);
        if (short_shadow_compare == "baseline" || short_shadow_compare == "both") {
            test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k,
                           rerank_candidates, query_mode, "baseline");
        }
        if (short_shadow_compare == "shadow" || short_shadow_compare == "both") {
            shadow_config.short_shadow = true;
            appr_alg->setGraphTurboConfig(shadow_config);
            test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k,
                           rerank_candidates, query_mode, "short_shadow");
        }
        print_index_file_size(path_index, query_mode);
        return;
    }
    const string layout_compare = getenv_string("RABITQ_LAYOUT_COMPARE", "original");
    if (layout_compare != "original" && layout_compare != "bfs" && layout_compare != "both")
        throw runtime_error("RABITQ_LAYOUT_COMPARE must be original, bfs, or both");

    unique_ptr<RaBitQHierarchicalNSW> bfs_index;
    const string bfs_path = path_index_string + ".bfs";
    if (layout_compare == "bfs" || layout_compare == "both") {
        bfs_index.reset(new RaBitQHierarchicalNSW(
            vecdim, vecsize, centroid_count, M, efConstruction, random_seed, false,
            external_residual_storage, residual_config));
        bfs_index->space().set_code_layout(code_layout);
        if (exists_test(bfs_path) && exists_test(quantizer_state_path(bfs_path)) &&
            exists_test(residual_state_path(bfs_path, query_mode))) {
            cout << "Loading BFS-reordered index from " << bfs_path << ":\n";
            bfs_index->loadIndex(bfs_path, vecsize);
        } else {
            cout << "Building BFS-reordered index from original layout:\n";
            StopW reorder_timer;
            bfs_index->importBfsReorderedFrom(
                *appr_alg, residual_state_path(bfs_path, query_mode));
            bfs_index->saveIndex(bfs_path);
            cout << "build_stage=bfs_reorder us=" << reorder_timer.getElapsedTimeMicro() << "\n";
        }
        cout << "bfs_reorder_validation"
             << " graph_fingerprint_original=" << appr_alg->labelGraphFingerprint()
             << " graph_fingerprint_bfs=" << bfs_index->labelGraphFingerprint()
             << " payload_fingerprint_original=" << appr_alg->payloadFingerprintByLabel()
             << " payload_fingerprint_bfs=" << bfs_index->payloadFingerprintByLabel()
             << " residual_fingerprint_original=" << appr_alg->residualFingerprintByLabel()
             << " residual_fingerprint_bfs=" << bfs_index->residualFingerprintByLabel()
             << " average_neighbor_id_distance_original=" << appr_alg->averageLevel0NeighborIdDistance()
             << " average_neighbor_id_distance_bfs=" << bfs_index->averageLevel0NeighborIdDistance()
             << "\n";
        if (appr_alg->labelGraphFingerprint() != bfs_index->labelGraphFingerprint() ||
            appr_alg->payloadFingerprintByLabel() != bfs_index->payloadFingerprintByLabel() ||
            appr_alg->residualFingerprintByLabel() != bfs_index->residualFingerprintByLabel())
            throw runtime_error("BFS-reordered index verification failed");
    }
    if (layout_compare == "original" || layout_compare == "both") {
        test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k,
                       rerank_candidates, query_mode, "original");
    }
    if (layout_compare == "bfs" || layout_compare == "both") {
        test_vs_recall(massQ, qsize, *bfs_index, path_data, vecdim, answers, k,
                       rerank_candidates, query_mode, "bfs");
    }
    print_index_file_size(path_index, query_mode);
}
