#pragma once

#include <algorithm>
#include <fstream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//hnsw底层算法和数据结构的定义，包含了空间接口、算法接口、访问列表池等核心组件的实现
#include "hnswlib.h"
#include "space_rabitq.h"

namespace hnswlib {

//输入原始向量，内部自动编码，再交给HNSW插入或搜索
class RaBitQHierarchicalNSW {
 private:
    static std::string quantizerStatePath(const std::string &location) {
        return location + ".rabitq";
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
        bool allow_replace_deleted = false)
        //构造spcae_和index_
        //ef_construction指的是建图候选数，M指的是最大邻居数
        : space_(dim, centroid_count, static_cast<uint32_t>(random_seed)),
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
    }
//添加数据点，首先将原始向量编码成RaBitQ编码，然后调用index_的addPoint方法插入编码后的向量
    void addPoint(const float *raw_vector, labeltype label, bool replace_deleted = false) {
        std::vector<char> encoded = space_.encodeVector(raw_vector);
        index_.addPoint(encoded.data(), label, replace_deleted);
    }
//查询向量不会编码成 database long code；查询只进行中心化和随机旋转。
//HNSW 使用 1-bit short code 计算距离下界，只有无法剪枝的节点才计算 8-bit long-code 距离。
//HNSW 导航、候选排序和最终结果始终使用 long distance。
    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(const float *raw_query, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
        return index_.searchKnn(raw_query, k, isIdAllowed);
    }
//搜索k近邻，返回结果按照距离从近到远排序，参数同上
    std::vector<std::pair<float, labeltype>>
    searchKnnCloserFirst(const float *raw_query, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
        return index_.searchKnnCloserFirst(raw_query, k, isIdAllowed);
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

    std::vector<std::pair<float, labeltype>>
    longCodeRerankCandidates(
        const float *raw_query,
        const std::vector<labeltype> &candidate_ids,
        size_t k) const {
        RaBitQSpace &mutable_space = const_cast<RaBitQSpace &>(space_);
        const void *query_context = mutable_space.prepare_query(raw_query);

        std::vector<std::pair<float, labeltype>> results;
        results.reserve(candidate_ids.size());
        try {
            for (labeltype label : candidate_ids) {
                tableint internal_id = 0;
                if (label < index_.cur_element_count.load(std::memory_order_relaxed) &&
                    index_.getExternalLabel(static_cast<tableint>(label)) == label) {
                    internal_id = static_cast<tableint>(label);
                } else {
                    std::lock_guard<std::mutex> lock(index_.label_lookup_lock);
                    auto it = index_.label_lookup_.find(label);
                    if (it == index_.label_lookup_.end()) {
                        continue;
                    }
                    internal_id = it->second;
                }
                const char *encoded = index_.getDataByInternalId(internal_id);
                results.emplace_back(mutable_space.query_distance(query_context, encoded), label);
            }
        } catch (...) {
            mutable_space.release_query(query_context);
            throw;
        }
        mutable_space.release_query(query_context);

        if (results.size() > k) {
            std::partial_sort(results.begin(), results.begin() + k, results.end());
            results.resize(k);
        } else {
            std::sort(results.begin(), results.end());
        }
        return results;
    }
};

}  // namespace hnswlib
