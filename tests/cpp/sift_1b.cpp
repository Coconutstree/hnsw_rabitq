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
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"

using namespace std;
using namespace hnswlib;

namespace {

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
    if (value == "residual4" || value == "residual8" ||
        value == "primary4_residual4" || value == "primary4_residual8") {
        return QueryMode::ResidualRerank;
    }
    throw runtime_error(
        "RABITQ_RERANK_MODE only supports residual4/residual8");
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
    bool external_residual_storage) {
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
    cout << "  build_distance=float32_l2"
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
        efs = {200};
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
    const string abc_ablation = getenv_string("RABITQ_ABC_ABLATION", "");
    if (!abc_ablation.empty() && abc_ablation != "A" &&
        abc_ablation != "B" && abc_ablation != "C")
        throw runtime_error("RABITQ_ABC_ABLATION must be A, B, or C");
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

    const string paper_data_dir = "/home/lyx_20251022/hnsw_rabitq/dbpedia_1M";
    const string dataset_name_config = getenv_string(
        "RABITQ_DATASET", paper_prune_profile ? "dbpedia_openai1536" : "sift10m");
    const DatasetConfig dataset{
        dataset_name_config,
        getenv_size_t("RABITQ_DIM", paper_prune_profile ? 1536 : 128),
        getenv_size_t("RABITQ_GT_WIDTH", paper_prune_profile ? 10 : 1000),
        getenv_string(
            "RABITQ_BASE_PATH",
            paper_prune_profile
                ? paper_data_dir + "/dbpedia_openai1536_base.fvecs"
                : "/home/kai3/coco/data/sift10m/sift10m_base.fvecs"),
        getenv_string(
            "RABITQ_QUERY_PATH",
            paper_prune_profile
                ? paper_data_dir + "/dbpedia_openai1536_query.fvecs"
                : "/home/kai3/coco/data/sift10m/sift10m_query.fvecs"),
        getenv_string(
            "RABITQ_GT_PATH",
            paper_prune_profile
                ? paper_data_dir + "/dbpedia_openai1536_groundtruth.ivecs"
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
        "%s_primary4_residual%zu_trueK%d_%s_%s_%s%s_floatbuild_ef_%d_M_%d.bin",
        dataset.index_prefix.c_str(),
        residual_bits,
        centroid_count,
        scale_name,
        scale_storage_name,
        code_layout_name.c_str(),
        block_suffix.c_str(),
        efConstruction,
        M);
    const string default_index_dir = paper_prune_profile
        ? "/home/lyx_20251022/hnsw_rabitq/build/indexes/dbpedia_ablation"
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
             << " explicit_environment_overrides=enabled\n";
    }

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
        if (exists_test(centroid_path)) {
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

        cout << "Building HNSW graph with float32 L2 distances, then encoding payloads to 4-bit RaBitQ\n";
        L2Space float_space(vecdim);
        const string float_graph_path = getenv_string(
            "RABITQ_FLOAT_GRAPH_PATH",
            join_path(index_dir, dataset.name + "_float_ef_" +
                std::to_string(efConstruction) + "_M_" + std::to_string(M) + ".bin"));
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
        if (!reuse_float_graph) {
            float_index->addPoint(first.data(), (size_t)0);
        }

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
            if (!reuse_float_graph) {
                float_index->addPoint(local_mass.data(), (size_t)label);
            }
        }
        const double payload_encode_wall_us = payload_encode_wall_timer.getElapsedTimeMicro();

        input.close();
        const double float_graph_build_us = float_graph_timer.getElapsedTimeMicro();
        if (!reuse_float_graph) {
            float_index->saveIndex(float_graph_path);
            cout << "Saved shared Float32 graph to " << float_graph_path << "\n";
        }
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
        cout << "build_total_us=" << total_build_timer.getElapsedTimeMicro() << "\n";
        print_index_file_size(path_index, query_mode);
    }

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
