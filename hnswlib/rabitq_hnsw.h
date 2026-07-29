#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

//hnsw底层算法和数据结构的定义，包含了空间接口、算法接口、访问列表池等核心组件的实现
#include "hnswlib.h"
#include "space_l2.h"
#include "space_rabitq.h"

namespace hnswlib {

//输入原始向量，内部自动编码，再交给HNSW插入或搜索
class RaBitQHierarchicalNSW {
 private:
    static std::string quantizerStatePath(const std::string &location) {
        return location + ".rabitq";
    }

    static std::string residualPathForIndex(const std::string &location) {
        if (location.find("nested4x4") != std::string::npos) {
            return location + ".nested4x4_low4";
        }
        return location + ".residual";
    }

 //RabitQ空间对象，负责将原始向量编码成RaBitQ编码，以及计算两个编码向量之间的距离
    RaBitQSpace space_;
    //负责对“编码后的向量”进行索引和搜索
    HierarchicalNSW<float> index_;
 public:
    //构造函数，接受维度、索引最大容量、质心数量、构图参数、随机种子和是否允许替换已删除元素等参数，初始化RaBitQ空间和HNSW索引
    RaBitQHierarchicalNSW(
        size_t dim,//原始向量的维度
        size_t max_elements,//索引的最大容量
        size_t centroid_count = 1,//RaBitQ编码使用的质心数量，默认为1
        size_t M = 16,//HNSW构图参数，节点的最大邻居数量，默认为16
        size_t ef_construction = 200,//HNSW构图参数，构图时考虑的候选节点数量，默认为200
        size_t random_seed = 100,//随机种子，用于生成随机旋转矩阵，默认为100
        //是否允许替换已删除元素，默认为false，如果为true，在插入新元素时会尝试替换掉被标记为删除的元素，以节省空间
        bool allow_replace_deleted = false,
        bool external_residual_storage = false,
        size_t residual_bits = 0)
        //构造spcae_和index_
        //ef_construction指的是建图候选数，M指的是最大邻居数
        : space_(
              dim,
              centroid_count,
              static_cast<uint32_t>(random_seed),
              external_residual_storage,
              residual_bits),
          index_(&space_, max_elements, M, ef_construction, random_seed, allow_replace_deleted) {
    }

    RaBitQHierarchicalNSW(
        size_t dim,
        size_t max_elements,
        size_t centroid_count,
        size_t M,
        size_t ef_construction,
        size_t random_seed,
        bool allow_replace_deleted,
        bool external_residual_storage,
        RaBitQSpace::ResidualQuantizationConfig residual_config)
        : space_(
              dim,
              centroid_count,
              static_cast<uint32_t>(random_seed),
              external_residual_storage,
              residual_config),
          index_(&space_, max_elements, M, ef_construction, random_seed, allow_replace_deleted) {
    }
//访问space_的接口，外部可以通过它改space_的参数或者调用space_的方法
    RaBitQSpace &space() {
        return space_;
    }

    const RaBitQSpace &space() const {
        return space_;
    }

    HierarchicalNSW<float> &index() {
        return index_;
    }

    const HierarchicalNSW<float> &index() const {
        return index_;
    }
//设置查询时考虑的候选节点数量，直接调用index_的setEf方法
    void setEf(size_t ef) {
        index_.setEf(ef);
    }
//保存索引到文件，直接调用index_的saveIndex方法
    void saveIndex(const std::string &location) {
        //这里保存的是hnsw索引里的编码数据，而不是原始的float向量
        std::ofstream quantizer_output(quantizerStatePath(location), std::ios::binary);
        if (!quantizer_output.is_open()) {
            throw std::runtime_error("RaBitQHierarchicalNSW failed to open quantizer state file for writing");
        }
        space_.saveState(quantizer_output);
        quantizer_output.close();

        index_.saveIndex(location);
    }
//加载索引
    void loadIndex(const std::string &location, size_t max_elements = 0) {
        std::ifstream quantizer_input(quantizerStatePath(location), std::ios::binary);
        if (!quantizer_input.is_open()) {
            throw std::runtime_error(
                "RaBitQHierarchicalNSW missing quantizer state file; expected " + quantizerStatePath(location));
        }
        space_.loadState(quantizer_input);
        quantizer_input.close();
        index_.loadIndex(location, &space_, max_elements);
        space_.openExternalResidualStorage(residualPathForIndex(location), index_.cur_element_count);
    }
//添加数据点，首先将原始向量编码成RaBitQ编码，然后调用index_的addPoint方法插入编码后的向量
    void addPoint(const float *raw_vector, labeltype label, bool replace_deleted = false) {
        std::vector<char> encoded = space_.encodeVector(raw_vector);
        index_.addPoint(encoded.data(), label, replace_deleted);
    }

    void importGraphFromFloatIndexWithPayloads(
        const HierarchicalNSW<float> &float_index,
        const std::vector<char> &payloads,
        size_t record_size,
        bool verbose = false) {
        const size_t total_count = float_index.cur_element_count;
        if (record_size != space_.get_data_size()) {
            throw std::runtime_error("RaBitQ payload record size does not match space data size");
        }
        if (payloads.size() != total_count * record_size) {
            throw std::runtime_error("RaBitQ payload array size does not match graph element count");
        }

        size_t copied_count = 0;
        const auto start_time = std::chrono::steady_clock::now();
        index_.importGraphAndCopyDataFrom(
            float_index,
            [this, &float_index, &payloads, record_size, total_count, verbose, start_time, &copied_count](
                tableint source_internal_id,
                void *target_data) {
                const labeltype label = float_index.getExternalLabel(source_internal_id);
                if (label >= total_count) {
                    throw std::runtime_error("RaBitQ payload label is outside payload array range");
                }
                std::memcpy(target_data, payloads.data() + label * record_size, record_size);
                ++copied_count;
                if (verbose && (copied_count % 100000 == 0 || copied_count == total_count)) {
                    const auto now = std::chrono::steady_clock::now();
                    const double seconds =
                        std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                    const double kips = seconds > 0.0 ? copied_count / (1000.0 * seconds) : 0.0;
                    std::cout << "Graph/payload import " << copied_count / (0.01 * total_count)
                              << " %, " << kips << " kips\n";
                }
            });
    }

    void importGraphFromFloatIndexWithPayloadFile(
        const HierarchicalNSW<float> &float_index,
        const std::string &payload_path,
        size_t record_size,
        bool verbose = false) {
        const size_t total_count = float_index.cur_element_count;
        if (record_size != space_.get_data_size()) {
            throw std::runtime_error("RaBitQ payload record size does not match space data size");
        }

        std::ifstream payload_input(payload_path.c_str(), std::ios::binary | std::ios::ate);
        if (!payload_input.is_open()) {
            throw std::runtime_error("RaBitQ failed to open payload file: " + payload_path);
        }
        const size_t payload_bytes = static_cast<size_t>(payload_input.tellg());
        if (payload_bytes != total_count * record_size) {
            throw std::runtime_error("RaBitQ payload file size does not match graph element count");
        }

        size_t copied_count = 0;
        const auto start_time = std::chrono::steady_clock::now();
        index_.importGraphAndCopyDataFrom(
            float_index,
            [this, &float_index, &payload_input, record_size, total_count, verbose, start_time, &copied_count](
                tableint source_internal_id,
                void *target_data) {
                const labeltype label = float_index.getExternalLabel(source_internal_id);
                if (label >= total_count) {
                    throw std::runtime_error("RaBitQ payload label is outside payload file range");
                }
                payload_input.clear();
                payload_input.seekg(static_cast<std::streamoff>(label * record_size), std::ios::beg);
                payload_input.read(
                    reinterpret_cast<char *>(target_data),
                    static_cast<std::streamsize>(record_size));
                if (!payload_input.good()) {
                    throw std::runtime_error("RaBitQ failed to read payload record from disk");
                }
                ++copied_count;
                if (verbose && (copied_count % 100000 == 0 || copied_count == total_count)) {
                    const auto now = std::chrono::steady_clock::now();
                    const double seconds =
                        std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                    const double kips = seconds > 0.0 ? copied_count / (1000.0 * seconds) : 0.0;
                    std::cout << "Graph/payload import " << copied_count / (0.01 * total_count)
                              << " %, " << kips << " kips\n";
                }
            });
    }

    void importGraphFromFloatIndexWithFullPayloadFileAndExternalResiduals(
        const HierarchicalNSW<float> &float_index,
        const std::string &payload_path,
        const std::string &residual_path,
        size_t full_record_size,
        bool verbose = false) {
        const size_t total_count = float_index.cur_element_count;
        if (!space_.external_residual_storage_enabled()) {
            throw std::runtime_error("RaBitQ external residual storage is disabled");
        }
        if (full_record_size != space_.get_full_data_size()) {
            throw std::runtime_error("RaBitQ full payload record size does not match space full data size");
        }

        std::ifstream payload_input(payload_path.c_str(), std::ios::binary | std::ios::ate);
        if (!payload_input.is_open()) {
            throw std::runtime_error("RaBitQ failed to open payload file: " + payload_path);
        }
        const size_t payload_bytes = static_cast<size_t>(payload_input.tellg());
        if (payload_bytes != total_count * full_record_size) {
            throw std::runtime_error("RaBitQ full payload file size does not match graph element count");
        }

        const int residual_fd = ::open(residual_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
        if (residual_fd < 0) {
            throw std::runtime_error("RaBitQ failed to create residual file: " + residual_path);
        }
        const size_t residual_total_bytes = total_count * space_.get_residual_disk_record_bytes();
        if (::ftruncate(residual_fd, static_cast<off_t>(residual_total_bytes)) != 0) {
            ::close(residual_fd);
            throw std::runtime_error("RaBitQ failed to resize residual file: " + residual_path);
        }

        size_t copied_count = 0;
        const auto start_time = std::chrono::steady_clock::now();
        std::vector<char> full_record(full_record_size, 0);
        std::vector<char> compact_record(space_.get_data_size(), 0);
        std::vector<char> residual_record(space_.get_residual_disk_record_bytes(), 0);
        index_.importGraphAndCopyDataFrom(
            float_index,
            [this,
             &float_index,
             &payload_input,
             &full_record,
             &compact_record,
             &residual_record,
             residual_fd,
             full_record_size,
             total_count,
             verbose,
             start_time,
             &copied_count,
             &residual_path](
                tableint source_internal_id,
                void *target_data) {
                const labeltype label = float_index.getExternalLabel(source_internal_id);
                if (label >= total_count) {
                    throw std::runtime_error("RaBitQ payload label is outside full payload file range");
                }
                payload_input.clear();
                payload_input.seekg(static_cast<std::streamoff>(label * full_record_size), std::ios::beg);
                payload_input.read(full_record.data(), static_cast<std::streamsize>(full_record_size));
                if (!payload_input.good()) {
                    throw std::runtime_error("RaBitQ failed to read full payload record from disk");
                }
                space_.copyCompactPayloadFromFull(full_record.data(), compact_record.data());
                std::memcpy(target_data, compact_record.data(), compact_record.size());
                space_.copyResidualRecordFromFull(full_record.data(), residual_record.data());

                size_t written = 0;
                const size_t record_size = residual_record.size();
                const off_t base_offset = static_cast<off_t>(label * record_size);
                while (written < record_size) {
                    const ssize_t n = ::pwrite(
                        residual_fd,
                        residual_record.data() + written,
                        record_size - written,
                        base_offset + static_cast<off_t>(written));
                    if (n < 0) {
                        if (errno == EINTR) {
                            continue;
                        }
                        throw std::runtime_error("RaBitQ failed to write residual record: " + residual_path);
                    }
                    if (n == 0) {
                        throw std::runtime_error("RaBitQ short write to residual file: " + residual_path);
                    }
                    written += static_cast<size_t>(n);
                }

                ++copied_count;
                if (verbose && (copied_count % 100000 == 0 || copied_count == total_count)) {
                    const auto now = std::chrono::steady_clock::now();
                    const double seconds =
                        std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                    const double kips = seconds > 0.0 ? copied_count / (1000.0 * seconds) : 0.0;
                    std::cout << "Graph/payload import " << copied_count / (0.01 * total_count)
                              << " %, " << kips << " kips\n";
                }
            });
        ::close(residual_fd);
        space_.openExternalResidualStorage(residual_path, total_count);
    }
//搜索k近邻，首先将查询向量编码成RaBitQ编码，然后调用index_的searchKnn方法搜索编码后的向量，返回距离和标签的二元组优先队列
    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(const float *raw_query, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
        return index_.searchKnn(raw_query, k, isIdAllowed);
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnnPlainThenResidualRerank(
        const float *raw_query,
        size_t k,
        size_t rerank_candidates,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        return index_.searchKnnPlainThenResidualRerank(raw_query, k, rerank_candidates, isIdAllowed);
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnnNested4x4Rerank(
        const float *raw_query,
        size_t k,
        size_t rerank_candidates,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        if (space_.get_residual_bits() != 4) {
            throw std::runtime_error("nested4x4 rerank requires 4-bit residual storage");
        }
        return index_.searchKnnNested4x4Rerank(
            raw_query,
            k,
            rerank_candidates,
            isIdAllowed);
    }
//搜索k近邻，返回结果按照距离从近到远排序，参数同上
    std::vector<std::pair<float, labeltype>>
    searchKnnCloserFirst(const float *raw_query, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
        return index_.searchKnnCloserFirst(raw_query, k, isIdAllowed);
    }

    std::vector<std::pair<float, labeltype>>
    searchKnnPlainThenResidualRerankCloserFirst(
        const float *raw_query,
        size_t k,
        size_t rerank_candidates,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        auto result = searchKnnPlainThenResidualRerank(raw_query, k, rerank_candidates, isIdAllowed);
        std::vector<std::pair<float, labeltype>> sorted;
        sorted.reserve(result.size());
        while (!result.empty()) {
            sorted.push_back(result.top());
            result.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        return sorted;
    }

    std::vector<std::pair<float, labeltype>>
    searchKnnNested4x4RerankCloserFirst(
        const float *raw_query,
        size_t k,
        size_t rerank_candidates,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        auto result =
            searchKnnNested4x4Rerank(raw_query, k, rerank_candidates, isIdAllowed);
        std::vector<std::pair<float, labeltype>> sorted;
        sorted.reserve(result.size());
        while (!result.empty()) {
            sorted.push_back(result.top());
            result.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        return sorted;
    }

    std::vector<labeltype>
    searchCandidateIds(const float *raw_query, size_t candidate_count, BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::vector<std::pair<float, labeltype>> candidates =
            searchKnnCloserFirst(raw_query, candidate_count, isIdAllowed);
        std::vector<labeltype> ids;
        ids.reserve(candidates.size());
        for (const auto &candidate : candidates) {
            ids.push_back(candidate.second);
        }
        return ids;
    }
};

}  // namespace hnswlib
