#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/rabitq_hnsw.h"

using namespace std;
using namespace hnswlib;

namespace {

struct StopW {
    chrono::steady_clock::time_point start;

    StopW() : start(chrono::steady_clock::now()) {}

    double elapsed_us() const {
        return chrono::duration_cast<chrono::microseconds>(
            chrono::steady_clock::now() - start).count();
    }
};

struct QuantizerState {
    size_t dim{0};
    size_t code_dim{0};
    size_t residual_bits{0};
    size_t data_size{0};
    size_t residual_disk_record_bytes{0};
    size_t centroid_count{1};
    uint32_t random_seed{0};
    vector<float> centroids;
    vector<float> fht_signs;
};

struct CandidateDistances {
    labeltype label{0};
    float float_l2{0.0f};
    float primary{0.0f};
    float exact_residual{0.0f};
};

size_t getenv_size_t(const char *name, size_t default_value) {
    const char *value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    if (value[0] == '-') {
        throw runtime_error(string(name) + " must be non-negative");
    }
    char *end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

string getenv_string(const char *name, const string &default_value) {
    const char *value = getenv(name);
    return (value == nullptr || value[0] == '\0') ? default_value : string(value);
}

size_t fvec_count_from_file_size(const string &path, size_t dim) {
    ifstream input(path.c_str(), ios::binary | ios::ate);
    if (!input.is_open()) {
        throw runtime_error("cannot open fvec file: " + path);
    }
    const size_t bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int) + dim * sizeof(float);
    if (record_bytes == 0 || bytes % record_bytes != 0) {
        throw runtime_error("invalid fvec file size: " + path);
    }
    return bytes / record_bytes;
}

void read_fvec_at(ifstream &input, size_t id, size_t dim, vector<float> &out) {
    const size_t record_bytes = sizeof(int) + dim * sizeof(float);
    input.clear();
    input.seekg(static_cast<streamoff>(id * record_bytes), ios::beg);
    int stored_dim = 0;
    input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
    if (!input.good() || stored_dim != static_cast<int>(dim)) {
        throw runtime_error("failed to read fvec record");
    }
    input.read(reinterpret_cast<char *>(out.data()), static_cast<streamsize>(dim * sizeof(float)));
    if (!input.good()) {
        throw runtime_error("failed to read fvec payload");
    }
}

void read_ivecs(const string &path, size_t qsize, size_t width, vector<unsigned int> &gt) {
    gt.assign(qsize * width, 0);
    ifstream input(path.c_str(), ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open gt file: " + path);
    }
    for (size_t i = 0; i < qsize; ++i) {
        int stored_width = 0;
        input.read(reinterpret_cast<char *>(&stored_width), sizeof(stored_width));
        if (!input.good() || stored_width != static_cast<int>(width)) {
            throw runtime_error("invalid gt record");
        }
        input.read(
            reinterpret_cast<char *>(gt.data() + i * width),
            static_cast<streamsize>(width * sizeof(unsigned int)));
        if (!input.good()) {
            throw runtime_error("failed to read gt payload");
        }
    }
}

template <typename T>
void read_pod(ifstream &input, T &value) {
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input.good()) {
        throw runtime_error("failed to read quantizer state");
    }
}

QuantizerState read_quantizer_state(const string &path) {
    ifstream input(path.c_str(), ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open quantizer state: " + path);
    }
    char magic[8];
    input.read(magic, sizeof(magic));
    const string magic_value(magic, sizeof(magic));
    const bool multi_centroid_format = magic_value == "EXRBTQ31";
    const bool single_centroid_format = magic_value == "EXRBTQ30";
    if (!multi_centroid_format && !single_centroid_format) {
        throw runtime_error("Exact residual oracle requires non-nested EXRBTQ30/EXRBTQ31 quantizer state");
    }

    uint64_t dim = 0;
    uint64_t code_dim = 0;
    uint32_t total_bits = 0;
    uint32_t short_bits = 0;
    uint32_t remaining_bits = 0;
    uint64_t encoded_header_size = 0;
    uint64_t short_factor_size = 0;
    uint64_t residual_factor_size = 0;
    uint64_t residual_scale_bytes = 0;
    uint32_t residual_bits = 0;
    uint64_t full_code_bytes = 0;
    uint64_t residual_code_bytes = 0;
    uint64_t data_size = 0;
    uint64_t full_data_size = 0;
    uint64_t residual_disk_record_bytes = 0;
    uint8_t external_residual_storage = 0;
    uint64_t residual_block_size = 0;
    uint32_t flags = 0;
    uint64_t centroid_count = 1;

    read_pod(input, dim);
    read_pod(input, code_dim);
    read_pod(input, total_bits);
    read_pod(input, short_bits);
    read_pod(input, remaining_bits);
    read_pod(input, encoded_header_size);
    read_pod(input, short_factor_size);
    read_pod(input, residual_factor_size);
    read_pod(input, residual_scale_bytes);
    read_pod(input, residual_bits);
    read_pod(input, full_code_bytes);
    read_pod(input, residual_code_bytes);
    read_pod(input, data_size);
    read_pod(input, full_data_size);
    read_pod(input, residual_disk_record_bytes);
    read_pod(input, external_residual_storage);
    read_pod(input, residual_block_size);
    read_pod(input, flags);
    if (multi_centroid_format) {
        read_pod(input, centroid_count);
    }

    QuantizerState state;
    state.dim = static_cast<size_t>(dim);
    state.code_dim = static_cast<size_t>(code_dim);
    state.residual_bits = static_cast<size_t>(residual_bits);
    state.data_size = static_cast<size_t>(data_size);
    state.residual_disk_record_bytes = static_cast<size_t>(residual_disk_record_bytes);
    state.centroid_count = static_cast<size_t>(centroid_count);
    read_pod(input, state.random_seed);
    state.centroids.assign(state.centroid_count * state.dim, 0.0f);
    state.fht_signs.assign(state.code_dim, 0.0f);
    input.read(
        reinterpret_cast<char *>(state.centroids.data()),
        static_cast<streamsize>(state.centroids.size() * sizeof(float)));
    input.read(
        reinterpret_cast<char *>(state.fht_signs.data()),
        static_cast<streamsize>(state.fht_signs.size() * sizeof(float)));
    if (!input.good()) {
        throw runtime_error("failed to read quantizer state payload");
    }

    if (total_bits != 4 || short_bits != 1 || remaining_bits != 3 ||
        encoded_header_size != sizeof(RaBitQSpace::EncodedHeader) ||
        short_factor_size != sizeof(RaBitQSpace::ShortCodeFactors) ||
        state.residual_bits != 4 || external_residual_storage == 0) {
        throw runtime_error("unexpected quantizer config for primary4 exact residual oracle");
    }
    return state;
}

void hadamard(vector<float> &values) {
    const size_t n = values.size();
    for (size_t step = 1; step < n; step <<= 1U) {
        for (size_t block = 0; block < n; block += (step << 1U)) {
            for (size_t i = 0; i < step; ++i) {
                const float a = values[block + i];
                const float b = values[block + step + i];
                values[block + i] = a + b;
                values[block + step + i] = a - b;
            }
        }
    }
}

void rotate_centered(
    const float *raw,
    const QuantizerState &state,
    size_t centroid_id,
    vector<float> &rotated) {
    rotated.assign(state.code_dim, 0.0f);
    const float *centroid = state.centroids.data() + centroid_id * state.dim;
    for (size_t i = 0; i < state.dim; ++i) {
        rotated[i] = (raw[i] - centroid[i]) * state.fht_signs[i];
    }
    hadamard(rotated);
}

uint8_t code_value(const uint8_t *packed_code, size_t index) {
    const uint8_t byte = packed_code[index >> 1U];
    return static_cast<uint8_t>((index & 1U) ? (byte >> 4U) : (byte & 0x0FU));
}

float l2_sqr(const float *lhs, const float *rhs, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

double topk_overlap(vector<pair<float, labeltype>> lhs, vector<pair<float, labeltype>> rhs, size_t k) {
    sort(lhs.begin(), lhs.end());
    sort(rhs.begin(), rhs.end());
    unordered_set<labeltype> set;
    for (size_t i = 0; i < k && i < lhs.size(); ++i) {
        set.insert(lhs[i].second);
    }
    size_t hits = 0;
    for (size_t i = 0; i < k && i < rhs.size(); ++i) {
        if (set.find(rhs[i].second) != set.end()) {
            hits++;
        }
    }
    return k == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace

int main() {
    cout.setf(ios::unitbuf);
    cerr.setf(ios::unitbuf);

    const size_t dim = 128;
    const size_t gt_width = 1000;
    const int ef_construction = 400;
    const int M = 32;
    const int centroid_count = static_cast<int>(getenv_size_t("RABITQ_CENTROID_COUNT", 64));
    const int random_seed = 100;
    const size_t residual_bits = 4;
    const size_t ef = getenv_size_t("RABITQ_EXACT_RESIDUAL_EF", 400);
    const size_t candidate_k = getenv_size_t("RABITQ_EXACT_RESIDUAL_CANDIDATES", 100);
    const size_t recall_k = getenv_size_t("RABITQ_EXACT_RESIDUAL_RECALL_K", 10);
    const size_t query_limit_env = getenv_size_t("RABITQ_EXACT_RESIDUAL_QUERIES", 1000);

    const string base_path = getenv_string(
        "RABITQ_BASE_PATH",
        "/home/kai3/coco/data/sift10m/sift10m_base.fvecs");
    const string query_path = getenv_string(
        "RABITQ_QUERY_PATH",
        "/home/kai3/coco/data/sift10m/sift10m_query.fvecs");
    const string gt_path = getenv_string(
        "RABITQ_GT_PATH",
        "/home/kai3/coco/data/sift10m/sift10m_groundtruth.ivecs");
    const string index_path = getenv_string(
        "RABITQ_INDEX_PATH",
        "build/sift10m_primary4_residual4_floatbuild_ef_400_M_32_C_64.bin");

    const size_t base_count = fvec_count_from_file_size(base_path, dim);
    const size_t query_count = fvec_count_from_file_size(query_path, dim);
    const size_t query_limit = min(query_count, query_limit_env);
    const QuantizerState state = read_quantizer_state(index_path + ".rabitq");
    if (state.centroid_count != static_cast<size_t>(centroid_count)) {
        throw runtime_error("RABITQ_CENTROID_COUNT does not match quantizer state");
    }

    cout << "Exact residual oracle config:\n";
    cout << "  dataset=sift10m"
         << " base_count=" << base_count
         << " query_count=" << query_limit
         << " dimension=" << dim << "\n";
    cout << "  M=" << M
         << " efConstruction=" << ef_construction
         << " efSearch=" << ef
         << " primary_top_candidates=" << candidate_k
         << " recall_at=" << recall_k << "\n";
    cout << "  primary_bits=4 residual_oracle=exact_unquantized"
         << " residual_bits_for_loaded_index=" << residual_bits
         << " centroid_count=" << centroid_count
         << " random_seed=" << random_seed << "\n";
    cout << "  index_path=" << index_path << "\n";
    cout << "  base_path=" << base_path << "\n";
    cout << "  query_path=" << query_path << "\n";
    cout << "  gt_path=" << gt_path << "\n";

    vector<float> queries(query_count * dim, 0.0f);
    ifstream query_input(query_path.c_str(), ios::binary);
    if (!query_input.is_open()) {
        throw runtime_error("cannot open query file: " + query_path);
    }
    vector<float> temp(dim, 0.0f);
    for (size_t i = 0; i < query_count; ++i) {
        read_fvec_at(query_input, i, dim, temp);
        memcpy(queries.data() + i * dim, temp.data(), dim * sizeof(float));
    }
    query_input.close();

    vector<unsigned int> gt;
    read_ivecs(gt_path, query_count, gt_width, gt);

    RaBitQHierarchicalNSW index(
        dim,
        base_count,
        centroid_count,
        M,
        ef_construction,
        random_seed,
        false,
        true,
        residual_bits);
    index.loadIndex(index_path, base_count);
    index.setEf(ef);

    vector<tableint> label_to_internal(base_count, 0);
    for (tableint internal_id = 0; internal_id < index.index().cur_element_count; ++internal_id) {
        const labeltype label = index.index().getExternalLabel(internal_id);
        if (label >= label_to_internal.size()) {
            throw runtime_error("index label is outside base vector range");
        }
        label_to_internal[label] = internal_id;
    }

    ifstream base_input(base_path.c_str(), ios::binary);
    if (!base_input.is_open()) {
        throw runtime_error("cannot open base file: " + base_path);
    }

    size_t total_gt_hits_in_primary_top = 0;
    size_t primary_recall_hits = 0;
    size_t exact_recall_hits = 0;
    size_t pair_count = 0;
    double primary_abs_error = 0.0;
    double exact_abs_error = 0.0;
    double primary_sq_error = 0.0;
    double exact_sq_error = 0.0;
    double max_primary_abs_error = 0.0;
    double max_exact_abs_error = 0.0;
    double top10_overlap_exact_float = 0.0;
    double manual_primary_vs_library_abs_error = 0.0;
    double max_manual_primary_vs_library_abs_error = 0.0;
    double query_rotation_abs_error = 0.0;
    size_t label_internal_mismatches = 0;
    double search_us = 0.0;
    double oracle_us = 0.0;

    vector<float> base_vector(dim, 0.0f);
    vector<float> rotated_base;
    vector<CandidateDistances> candidates;

    for (size_t query_id = 0; query_id < query_limit; ++query_id) {
        StopW search_timer;
        priority_queue<pair<float, labeltype>> primary_queue =
            index.index().searchKnn(queries.data() + query_id * dim, candidate_k);
        search_us += search_timer.elapsed_us();

        candidates.clear();
        candidates.reserve(primary_queue.size());
        while (!primary_queue.empty()) {
            CandidateDistances candidate;
            candidate.primary = primary_queue.top().first;
            candidate.label = primary_queue.top().second;
            primary_queue.pop();
            candidates.push_back(candidate);
        }

        unordered_set<labeltype> gt_top_recall;
        unordered_set<labeltype> candidate_labels;
        for (size_t i = 0; i < recall_k && i < gt_width; ++i) {
            gt_top_recall.insert(gt[query_id * gt_width + i]);
        }
        for (const auto &candidate : candidates) {
            candidate_labels.insert(candidate.label);
        }
        for (size_t i = 0; i < recall_k && i < gt_width; ++i) {
            if (candidate_labels.find(gt[query_id * gt_width + i]) != candidate_labels.end()) {
                total_gt_hits_in_primary_top++;
            }
        }

        const auto *prepared_query =
            static_cast<const RaBitQSpace::PreparedQuery *>(
                index.space().prepare_query(queries.data() + query_id * dim));

        StopW oracle_timer;
        vector<pair<float, labeltype>> float_rank;
        vector<pair<float, labeltype>> primary_rank;
        vector<pair<float, labeltype>> exact_rank;
        float_rank.reserve(candidates.size());
        primary_rank.reserve(candidates.size());
        exact_rank.reserve(candidates.size());

        for (CandidateDistances &candidate : candidates) {
            const tableint internal_id = label_to_internal[candidate.label];
            if (index.index().getExternalLabel(internal_id) != candidate.label) {
                label_internal_mismatches++;
            }
            read_fvec_at(base_input, static_cast<size_t>(candidate.label), dim, base_vector);
            const char *encoded = index.index().getDataByInternalId(internal_id);
            RaBitQSpace::EncodedHeader header;
            memcpy(&header, encoded, sizeof(header));
            const uint8_t *primary_code =
                reinterpret_cast<const uint8_t *>(
                    encoded + sizeof(RaBitQSpace::EncodedHeader) +
                    sizeof(RaBitQSpace::ShortCodeFactors));
            const size_t centroid_id_offset =
                sizeof(RaBitQSpace::EncodedHeader) +
                sizeof(RaBitQSpace::ShortCodeFactors) +
                (state.code_dim + 1U) / 2U;
            const uint8_t centroid_id =
                *reinterpret_cast<const uint8_t *>(encoded + centroid_id_offset);
            if (static_cast<size_t>(centroid_id) >= prepared_query->centroid_queries.size()) {
                throw runtime_error("candidate centroid_id is out of range");
            }
            const RaBitQSpace::QueryContext *query_context =
                &prepared_query->centroid_queries[centroid_id];
            vector<float> rotated_query_check;
            rotate_centered(queries.data() + query_id * dim, state, centroid_id, rotated_query_check);
            rotate_centered(base_vector.data(), state, centroid_id, rotated_base);
            for (size_t d = 0; d < state.code_dim; ++d) {
                query_rotation_abs_error += abs(
                    static_cast<double>(rotated_query_check[d]) -
                    static_cast<double>(query_context->rotated_residual[d]));
            }

            double exact_ip = 0.0;
            double manual_primary_inner_twice = 0.0;
            for (size_t d = 0; d < state.code_dim; ++d) {
                const double decoded_primary =
                    0.5 * static_cast<double>(header.long_scale) *
                    (static_cast<double>(code_value(primary_code, d)) -
                     static_cast<double>(RaBitQSpace::kUnsignedOffset));
                const double e_true = static_cast<double>(rotated_base[d]) - decoded_primary;
                exact_ip += static_cast<double>(query_context->rotated_residual[d]) * e_true;
                manual_primary_inner_twice +=
                    2.0 * static_cast<double>(query_context->rotated_residual[d]) *
                    decoded_primary;
            }
            const double manual_primary_distance =
                static_cast<double>(query_context->query_norm_sqr) +
                static_cast<double>(header.norm_sqr) -
                manual_primary_inner_twice;
            const double manual_primary_error =
                manual_primary_distance - static_cast<double>(candidate.primary);
            manual_primary_vs_library_abs_error += abs(manual_primary_error);
            max_manual_primary_vs_library_abs_error =
                max(max_manual_primary_vs_library_abs_error, abs(manual_primary_error));

            candidate.float_l2 = l2_sqr(queries.data() + query_id * dim, base_vector.data(), dim);
            candidate.exact_residual = static_cast<float>(
                static_cast<double>(candidate.primary) - 2.0 * exact_ip);

            const double primary_error =
                static_cast<double>(candidate.primary) - candidate.float_l2;
            const double exact_error =
                static_cast<double>(candidate.exact_residual) - candidate.float_l2;
            primary_abs_error += abs(primary_error);
            exact_abs_error += abs(exact_error);
            primary_sq_error += primary_error * primary_error;
            exact_sq_error += exact_error * exact_error;
            max_primary_abs_error = max(max_primary_abs_error, abs(primary_error));
            max_exact_abs_error = max(max_exact_abs_error, abs(exact_error));
            pair_count++;

            float_rank.emplace_back(candidate.float_l2, candidate.label);
            primary_rank.emplace_back(candidate.primary, candidate.label);
            exact_rank.emplace_back(candidate.exact_residual, candidate.label);
        }
        oracle_us += oracle_timer.elapsed_us();
        index.space().release_query(prepared_query);

        sort(primary_rank.begin(), primary_rank.end());
        sort(exact_rank.begin(), exact_rank.end());
        for (size_t i = 0; i < recall_k && i < primary_rank.size(); ++i) {
            if (gt_top_recall.find(primary_rank[i].second) != gt_top_recall.end()) {
                primary_recall_hits++;
            }
            if (gt_top_recall.find(exact_rank[i].second) != gt_top_recall.end()) {
                exact_recall_hits++;
            }
        }
        top10_overlap_exact_float += topk_overlap(float_rank, exact_rank, recall_k);
    }

    const double pairs = static_cast<double>(max<size_t>(1, pair_count));
    const double queries_n = static_cast<double>(max<size_t>(1, query_limit));
    const double recall_denominator =
        static_cast<double>(query_limit * min(recall_k, gt_width));

    cout << fixed << setprecision(6);
    cout << "Exact residual oracle results:\n";
    cout << "  queries=" << query_limit
         << " pairs=" << pair_count
         << " primary_top_candidates=" << candidate_k
         << " recall_at=" << recall_k << "\n";
    cout << "  candidate_coverage@" << recall_k << "="
         << total_gt_hits_in_primary_top / recall_denominator << "\n";
    cout << "  primary_recall@" << recall_k << "="
         << primary_recall_hits / recall_denominator << "\n";
    cout << "  exact_residual_recall@" << recall_k << "="
         << exact_recall_hits / recall_denominator << "\n";
    cout << "  exact_top" << recall_k << "_overlap_with_float_oracle="
         << top10_overlap_exact_float / queries_n << "\n";
    cout << "  primary_vs_float_l2_mae=" << primary_abs_error / pairs
         << " rmse=" << sqrt(primary_sq_error / pairs)
         << " max_abs=" << max_primary_abs_error << "\n";
    cout << "  exact_residual_vs_float_l2_mae=" << exact_abs_error / pairs
         << " rmse=" << sqrt(exact_sq_error / pairs)
         << " max_abs=" << max_exact_abs_error << "\n";
    cout << "  self_check_manual_primary_vs_library_mae="
         << manual_primary_vs_library_abs_error / pairs
         << " max_abs=" << max_manual_primary_vs_library_abs_error
         << " query_rotation_mean_abs_error="
         << query_rotation_abs_error / (queries_n * static_cast<double>(state.code_dim))
         << " label_internal_mismatches=" << label_internal_mismatches << "\n";
    cout << "  avg_primary_search_us=" << search_us / queries_n
         << " avg_exact_oracle_compute_us=" << oracle_us / queries_n << "\n";
    return 0;
}
