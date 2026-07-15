#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

#ifndef HNSW_EF_CONSTRUCTION
#define HNSW_EF_CONSTRUCTION 200
#endif

namespace {

void print_run_config(
    const char *dataset_name,
    size_t vecsize,
    size_t qsize,
    size_t vecdim,
    int efConstruction,
    int M,
    const char *path_index,
    const char *path_data,
    const char *path_q,
    const char *path_gt) {
    cout << "Run config:\n";
    cout << "  dataset=" << dataset_name << "\n";
    cout << "  base_count=" << vecsize << "\n";
    cout << "  query_count=" << qsize << "\n";
    cout << "  dimension=" << vecdim << "\n";
    cout << "  M=" << M << " efConstruction=" << efConstruction << "\n";
    cout << "  quantizer=none raw_float_l2\n";
    cout << "  base_path=" << path_data << "\n";
    cout << "  query_path=" << path_q << "\n";
    cout << "  gt_path=" << path_gt << "\n";
    cout << "  index_path=" << path_index << "\n";
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

size_t file_size_bytes(const string &path) {
    ifstream input(path, ios::binary | ios::ate);
    if (!input.is_open()) {
        return 0;
    }
    return static_cast<size_t>(input.tellg());
}

void print_index_file_size(
    const string &index_path,
    size_t vecsize,
    size_t vecdim,
    size_t extra_rerank_bytes = 0) {
    const size_t index_bytes = file_size_bytes(index_path);
    const size_t auxiliary_bytes = 0;
    const size_t total_bytes = index_bytes + auxiliary_bytes;
    const size_t effective_query_storage_bytes = total_bytes + extra_rerank_bytes;
    const size_t raw_vector_bytes = vecsize * vecdim * sizeof(float);
    const size_t graph_and_metadata_bytes =
        total_bytes > raw_vector_bytes ? total_bytes - raw_vector_bytes : 0;
    const double mb = 1000000.0;
    cout << "Index storage size: " << total_bytes / mb << " MB"
         << " (index=" << index_bytes / mb << " MB"
         << ", auxiliary=" << auxiliary_bytes / mb << " MB"
         << ", total_bytes=" << total_bytes << ")\n";
    cout << "storage_breakdown"
         << " index_file_MB=" << index_bytes / mb
         << " auxiliary_MB=" << auxiliary_bytes / mb
         << " raw_vector_inside_index_MB=" << raw_vector_bytes / mb
         << " graph_and_metadata_estimate_MB=" << graph_and_metadata_bytes / mb
         << " extra_rerank_data_MB=" << extra_rerank_bytes / mb
         << " effective_query_storage_MB=" << effective_query_storage_bytes / mb
         << " effective_query_storage_bytes=" << effective_query_storage_bytes
         << "\n";
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
    double norm = 0.0;
    for (size_t i = 0; i < vecdim; ++i) {
        norm += static_cast<double>(dst[i]) * static_cast<double>(dst[i]);
    }
    norm = std::sqrt(norm);
    if (norm != 0.0) {
        for (size_t i = 0; i < vecdim; ++i) {
            dst[i] = static_cast<float>(static_cast<double>(dst[i]) / norm);
        }
    }
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
    const vector<unsigned int> &massQA,
    size_t qsize,
    size_t gt_topk,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<float, labeltype>>>(qsize)).swap(answers);
    cout << qsize << "\n";
    for (size_t i = 0; i < qsize; i++) {
        for (size_t j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_topk * i + j]);
        }
    }
}

struct SearchReport {
    float recall{0.0f};
    float hnsw_search_us_per_query{0.0f};
    float total_us_per_query{0.0f};
};

static SearchReport test_approx(
    float *massQ,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;
    double hnsw_us = 0.0;

    for (size_t i = 0; i < qsize; i++) {
        StopW hnsw_timer;
        std::priority_queue<std::pair<float, labeltype>> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        hnsw_us += hnsw_timer.getElapsedTimeMicro();

        std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (!gt.empty()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (!result.empty()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            }
            result.pop();
        }
    }

    SearchReport report;
    report.recall = total == 0 ? 0.0f : 1.0f * correct / total;
    report.hnsw_search_us_per_query = static_cast<float>(hnsw_us / static_cast<double>(qsize));
    report.total_us_per_query = report.hnsw_search_us_per_query;
    return report;
}

static void test_vs_recall(
    float *massQ,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
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
        SearchReport report = test_approx(massQ, qsize, appr_alg, vecdim, answers, k);

        cout << ef << "\t" << report.recall
             << "\t" << report.total_us_per_query << " us"
             << "\t" << "hnsw_search_us_per_query=" << report.hnsw_search_us_per_query
             << "\t" << "total_us_per_query=" << report.total_us_per_query << "\n";
        if (report.recall > 1.0f) {
            cout << report.recall << "\t" << report.total_us_per_query << " us\n";
            break;
        }
    }
}

void sift_test1B() {
    const char *dataset_name = "deep1B";
    const int efConstruction = HNSW_EF_CONSTRUCTION;
    const int M = 16;
    const size_t vecdim = 96;
    const size_t gt_topk = 100;

    char path_index[1024];
    const char *path_q = "/home/kai3/coco/data/deep1B/deep1B_query.fvecs";
    const char *path_data = "/home/kai3/coco/data/deep1B/deep1B_base.fvecs";
    const char *path_gt = "/home/kai3/coco/data/deep1B/deep1B_groundtruth.ivecs";
    const size_t vecsize = fvec_count_from_file_size(path_data, vecdim);
    const size_t qsize = fvec_count_from_file_size(path_q, vecdim);
    snprintf(
        path_index,
        sizeof(path_index),
        "deep1B_hnsw_ef_%d_M_%d.bin",
        efConstruction,
        M);

    print_run_config(
        dataset_name,
        vecsize,
        qsize,
        vecdim,
        efConstruction,
        M,
        path_index,
        path_data,
        path_q,
        path_gt);

    L2Space l2space(vecdim);
    cout << "  raw_bytes_per_vector=" << l2space.get_data_size()
         << " (float32 raw L2)\n";

    HierarchicalNSW<float> *appr_alg = nullptr;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, path_index, false);
        print_index_file_size(path_index, vecsize, vecdim);
    } else {
        ifstream input(path_data, ios::binary);
        if (!input.is_open()) {
            throw runtime_error("cannot open base file");
        }

        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);

        std::vector<float> first(vecdim, 0.0f);
        read_fvec_as_float(input, first.data(), vecdim);
        appr_alg->addPoint(first.data(), (size_t)0);

        int j1 = 0;
        StopW stopw;
        StopW stopw_full;
        const size_t report_every = 100000;

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
            appr_alg->addPoint(local_mass.data(), (size_t)label);
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
        print_index_file_size(path_index, vecsize, vecdim);
    }

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    if (!inputGT.is_open()) {
        throw runtime_error("cannot open gt file");
    }
    vector<unsigned int> massQA(qsize * gt_topk);
    for (size_t i = 0; i < qsize; i++) {
        int t = 0;
        inputGT.read(reinterpret_cast<char *>(&t), 4);
        if (!inputGT.good()) {
            throw runtime_error("gt file error");
        }
        if (t != static_cast<int>(gt_topk)) {
            throw runtime_error("gt file error");
        }
        inputGT.read(reinterpret_cast<char *>(massQA.data() + gt_topk * i), t * 4);
        if (!inputGT.good()) {
            throw runtime_error("gt file error");
        }
    }
    inputGT.close();

    cout << "Loading queries:\n";
    vector<float> massQ(qsize * vecdim);
    ifstream inputQ(path_q, ios::binary);
    if (!inputQ.is_open()) {
        throw runtime_error("cannot open query file");
    }
    for (size_t i = 0; i < qsize; i++) {
        read_fvec_as_float(inputQ, massQ.data() + i * vecdim, vecdim);
    }
    inputQ.close();

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    const size_t k = 1;
    cout << "Parsing gt:\n";
    get_gt(massQA, qsize, gt_topk, answers, k);
    cout << "Loaded gt\n";
    test_vs_recall(massQ.data(), qsize, *appr_alg, vecdim, answers, k);
    print_index_file_size(path_index, vecsize, vecdim);
}
