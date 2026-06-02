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

namespace {

void print_run_config(
    int subset_size_millions,
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
    cout << "Run config:\n";
    cout << "  dataset=BigANN/SIFT1B\n";
    cout << "  subset_size_millions=" << subset_size_millions
         << " (" << vecsize << " base vectors)\n";
    cout << "  query_count=" << qsize << "\n";
    cout << "  dimension=" << vecdim << "\n";
    cout << "  M=" << M << " efConstruction=" << efConstruction << "\n";
    cout << "  quantizer=RaBitQ centroid_count=" << centroid_count
         << " rerank_candidates=" << rerank_candidates
         << " random_seed=" << random_seed << "\n";
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
    size_t sample_count,
    vector<unsigned char> &scratch) {
    input.clear();
    input.seekg(0, ios::beg);

    const size_t actual_sample_count = min(sample_count, vecsize);
    if (actual_sample_count == 0) {
        throw runtime_error("not enough samples to train global center");
    }

    vector<float> center(vecdim, 0.0f);
    vector<float> sample(vecdim, 0.0f);
    for (size_t i = 0; i < actual_sample_count; ++i) {
        read_bvec_as_float(input, sample.data(), vecdim, scratch);
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
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<float, labeltype>>>(qsize)).swap(answers);
    cout << qsize << "\n";
    for (size_t i = 0; i < qsize; i++) {
        for (size_t j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

static float test_approx(
    float *massQ,
    size_t qsize,
    RaBitQHierarchicalNSW &appr_alg,
    const string &base_path,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    size_t rerank_candidates) {
    size_t correct = 0;
    size_t total = 0;
    vector<float> candidate(vecdim, 0.0f);
    BVecRandomReader base_reader(base_path, vecdim);

    for (size_t i = 0; i < qsize; i++) {
        vector<labeltype> candidate_ids = appr_alg.searchCandidateIds(massQ + vecdim * i, rerank_candidates);
        vector<pair<float, labeltype>> reranked;
        reranked.reserve(candidate_ids.size());
        for (labeltype label : candidate_ids) {
            base_reader.readVector(label, candidate.data());
            float dist = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                const float diff = massQ[vecdim * i + d] - candidate[d];
                dist += diff * diff;
            }
            reranked.emplace_back(dist, label);
        }
        if (reranked.size() > k) {
            partial_sort(reranked.begin(), reranked.begin() + k, reranked.end());
            reranked.resize(k);
        } else {
            sort(reranked.begin(), reranked.end());
        }
        std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (!gt.empty()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for (const auto &entry : reranked) {
            if (g.find(entry.second) != g.end()) {
                correct++;
            }
        }
    }
    return total == 0 ? 0.0f : 1.0f * correct / total;
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
    const size_t min_ef = std::max(k, rerank_candidates);
    for (size_t i = min_ef; i < 30; i++) {
        efs.push_back(i);
    }
    for (size_t i = std::max<size_t>(30, min_ef); i < 100; i += 10) {
        efs.push_back(i);
    }
    for (size_t i = std::max<size_t>(100, min_ef); i < 500; i += 40) {
        efs.push_back(i);
    }

    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw;

        float recall = test_approx(massQ, qsize, appr_alg, base_path, vecdim, answers, k, rerank_candidates);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0f) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

void sift_test1B() {
    const int subset_size_millions = 10;
    const int efConstruction = 40;
    const int M = 16;
    const int centroid_count = 64;
    const int rerank_candidates = 100;
    const size_t centroid_train_samples = 200000;
    const int random_seed = 100;

    const size_t vecsize = subset_size_millions * 1000000ULL;
    const size_t qsize = 10000;
    const size_t vecdim = 128;

    char path_index[1024];
    char path_gt[1024];
    const char *path_q = "/home/kai3/coco/hnswlib/bigann/bigann_query.bvecs";
    const char *path_data = "/home/kai3/coco/hnswlib/bigann/bigann_base.bvecs";
    snprintf(
        path_index,
        sizeof(path_index),
        "sift1b_rabitq_%dm_ef_%d_M_%d_C_%d.bin",
        subset_size_millions,
        efConstruction,
        M,
        centroid_count);
    snprintf(path_gt, sizeof(path_gt), "/home/kai3/coco/hnswlib/bigann/gnd/idx_%dM.ivecs", subset_size_millions);

    print_run_config(
        subset_size_millions,
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
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (size_t i = 0; i < qsize; i++) {
        int t = 0;
        inputGT.read((char *)&t, 4);
        if (!inputGT.good()) {
            throw runtime_error("gt file error");
        }
        inputGT.read((char *)(massQA + 1000 * i), t * 4);
        if (!inputGT.good() || t != 1000) {
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
    vector<unsigned char> scratch(vecdim);
    for (size_t i = 0; i < qsize; i++) {
        read_bvec_as_float(inputQ, massQ + i * vecdim, vecdim, scratch);
    }
    inputQ.close();

    ifstream input(path_data, ios::binary);
    if (!input.is_open()) {
        throw runtime_error("cannot open base file");
    }

    RaBitQHierarchicalNSW *appr_alg = new RaBitQHierarchicalNSW(
        vecdim, vecsize, centroid_count, M, efConstruction, random_seed);

    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        if (!exists_test(quantizer_state_path(path_index))) {
            throw runtime_error("missing RaBitQ quantizer state sidecar; rebuild the index to regenerate it");
        }
        appr_alg->loadIndex(path_index, vecsize);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        cout << "Training one global ExRaBitQ center from "
             << min(centroid_train_samples, vecsize) << " base vectors\n";
        vector<float> global_center = train_global_center(
            input,
            vecdim,
            vecsize,
            centroid_train_samples,
            scratch);
        appr_alg->space().setGlobalCenter(global_center.data());

        vector<float> first(vecdim);
        read_bvec_as_float(input, first.data(), vecdim, scratch);
        appr_alg->addPoint(first.data(), (size_t)0);

        int j1 = 0;
        StopW stopw;
        StopW stopw_full;
        const size_t report_every = 100000;

#pragma omp parallel for
        for (int i = 1; i < static_cast<int>(vecsize); i++) {
            float local_mass[128];
            int label = 0;
#pragma omp critical
            {
                read_bvec_as_float(input, local_mass, vecdim, scratch);
                j1++;
                label = j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                         << " Mem: " << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint(local_mass, (size_t)label);
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
    }

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    const size_t k = 1;
    cout << "Parsing gt:\n";
    get_gt(massQA, qsize, answers, k);
    cout << "Loaded gt\n";
    test_vs_recall(massQ, qsize, *appr_alg, path_data, vecdim, answers, k, rerank_candidates);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
}
