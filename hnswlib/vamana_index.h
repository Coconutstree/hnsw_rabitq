#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hnswlib.h"
#include "space_rabitq.h"
#include "vamana_prune.h"
#include "vamana_search.h"

namespace hnswlib {

class VamanaIndex {
 public:
    static constexpr size_t kDefaultR = 32;
    static constexpr size_t kDefaultLBuild = 400;
    static constexpr float kDefaultAlpha = 1.2f;
    static constexpr size_t kDefaultBeamWidth = 1;

    struct DegreeStats {
        size_t max_degree{0};
        double avg_degree{0.0};
    };
    using BuildProgressCallback = std::function<void(size_t, size_t)>;

    class BuildEvaluator {
     public:
        BuildEvaluator(RaBitQSpace &space, const VamanaIndex &index, const void *encoded_query)
            : space_(space), index_(index) {
            prepared_ = space_.prepare_symmetric_build_query(encoded_query);
        }

        BuildEvaluator(RaBitQSpace &space, const VamanaIndex &index, vamana::NodeId query_id)
            : BuildEvaluator(space, index, index.getDataByInternalId(query_id)) {}

        ~BuildEvaluator() {
            space_.release_symmetric_build_query(prepared_);
        }

        float distance(vamana::NodeId id) {
            return space_.symmetric_build_distance_prepared(
                prepared_, index_.getDataByInternalId(id));
        }

        void distance_batch(const vamana::NodeId *ids, size_t count, float *distances) {
            points_.resize(count);
            for (size_t i = 0; i < count; ++i) {
                points_[i] = index_.getDataByInternalId(ids[i]);
            }
            space_.symmetric_build_distance_batch(
                prepared_, points_.data(), count, distances, 2);
        }

     private:
        RaBitQSpace &space_;
        const VamanaIndex &index_;
        const void *prepared_{nullptr};
        std::vector<const void *> points_;
    };

    class QueryEvaluator {
     public:
        QueryEvaluator(RaBitQSpace &space, const VamanaIndex &index, const float *raw_query)
            : space_(space), index_(index) {
            prepared_ = space_.prepare_query(raw_query);
        }

        ~QueryEvaluator() {
            space_.release_query(prepared_);
        }

        float distance(vamana::NodeId id) {
            return space_.query_distance(prepared_, index_.getDataByInternalId(id));
        }

        PaperPruneEstimate<float> paper_estimate(vamana::NodeId id, float epsilon0) {
            if (index_.paperPruneSidecarReady()) {
                return space_.compute_paper_prune_estimate_sidecar(
                    prepared_,
                    index_.paperMsbCode(id),
                    index_.paperPruneFactors(id),
                    epsilon0);
            }
            return space_.compute_paper_prune_estimate(
                prepared_, index_.getDataByInternalId(id), epsilon0);
        }

        float distance_with_paper(vamana::NodeId id, float short_ip) {
            return space_.query_distance_with_paper_msb(
                prepared_, index_.getDataByInternalId(id), short_ip);
        }

        void prefetch(vamana::NodeId id) const {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(index_.getDataByInternalId(id), 0, 3);
#else
            (void) id;
#endif
        }

        void distance_batch(const vamana::NodeId *ids, size_t count, float *distances) {
            points_.resize(count);
            for (size_t i = 0; i < count; ++i) {
                points_[i] = index_.getDataByInternalId(ids[i]);
            }
            space_.query_distance_batch_k1(prepared_, points_.data(), count, distances, 2);
        }

        size_t active_centroids() const {
            return space_.query_active_centroids(prepared_);
        }

        const void *prepared_query() const {
            return prepared_;
        }

     private:
        RaBitQSpace &space_;
        const VamanaIndex &index_;
        const void *prepared_{nullptr};
        std::vector<const void *> points_;
    };

    VamanaIndex(
        RaBitQSpace *space,
        size_t max_elements,
        size_t r = kDefaultR,
        size_t l_build = kDefaultLBuild,
        float alpha = kDefaultAlpha,
        size_t beam_width = kDefaultBeamWidth)
        : space_(space),
          max_elements_(max_elements),
          r_(r),
          l_build_(l_build),
          alpha_(alpha),
          beam_width_(beam_width) {
        if (space_ == nullptr) {
            throw std::invalid_argument("VamanaIndex requires a RaBitQSpace");
        }
        if (r_ == 0 || l_build_ == 0 || beam_width_ == 0) {
            throw std::invalid_argument("VamanaIndex R, L_build, and beam_width must be positive");
        }
        data_size_ = space_->get_data_size();
        labels_.reserve(max_elements_);
        graph_.reserve(max_elements_);
    }

    size_t size() const { return cur_element_count_; }
    size_t max_elements() const { return max_elements_; }
    size_t data_size() const { return data_size_; }
    size_t r() const { return r_; }
    size_t l_build() const { return l_build_; }
    float alpha() const { return alpha_; }
    size_t beam_width() const { return beam_width_; }
    vamana::NodeId start_node() const { return start_node_; }
    bool paper_prune_active() const { return paper_prune_active_; }
    float paper_epsilon0() const { return paper_epsilon0_; }
    size_t paper_prefetch_distance() const { return paper_prefetch_distance_; }

    void setPaperPrune(bool active, float epsilon0 = 1.9f) {
        if (epsilon0 < 0.0f || !std::isfinite(epsilon0)) {
            throw std::invalid_argument("Vamana paper-prune epsilon0 must be finite and non-negative");
        }
        paper_prune_active_ = active;
        paper_epsilon0_ = epsilon0;
    }

    void preparePaperPruneSidecar() const {
        ensurePaperPruneSidecar();
    }

    const char *getDataByInternalId(vamana::NodeId id) const {
        if (id >= cur_element_count_) {
            throw std::out_of_range("Vamana internal id is outside current element count");
        }
        return payloads_.data() + static_cast<size_t>(id) * data_size_;
    }

    char *getDataByInternalId(vamana::NodeId id) {
        return const_cast<char *>(static_cast<const VamanaIndex *>(this)->getDataByInternalId(id));
    }

    labeltype getExternalLabel(vamana::NodeId id) const {
        if (id >= labels_.size()) {
            throw std::out_of_range("Vamana internal id has no label");
        }
        return labels_[id];
    }

    const std::vector<std::vector<vamana::NodeId>> &graph() const { return graph_; }

    void addPointEncodedSymmetric(const void *encoded_data, labeltype label) {
        if (encoded_data == nullptr) {
            throw std::invalid_argument("Vamana insertion received null encoded data");
        }
        if (cur_element_count_ >= max_elements_) {
            throw std::runtime_error("Vamana index capacity exceeded");
        }
        if (label_lookup_.find(label) != label_lookup_.end()) {
            throw std::runtime_error("Vamana index does not support updates for duplicate labels");
        }

        const vamana::NodeId id = static_cast<vamana::NodeId>(cur_element_count_);
        ensurePayloadStorage();
        std::memcpy(payloads_.data() + cur_element_count_ * data_size_, encoded_data, data_size_);
        labels_.push_back(label);
        label_lookup_[label] = id;
        graph_.push_back(std::vector<vamana::NodeId>());
        ++cur_element_count_;

        if (cur_element_count_ == 1) {
            start_node_ = id;
            return;
        }

        BuildEvaluator search_eval(*space_, *this, encoded_data);
        vamana::SearchStats ignored;
        vamana::SearchScratch search_scratch;
        const std::vector<vamana::SearchCandidate> &visited =
            vamana::search_vamana(graph_, start_node_, cur_element_count_ - 1U,
                                  l_build_, beam_width_, search_eval, search_scratch, &ignored);

        std::vector<vamana::PruneCandidate> pool;
        pool.reserve(visited.size());
        for (const vamana::SearchCandidate &candidate : visited) {
            pool.push_back(vamana::PruneCandidate{candidate.distance, candidate.id});
        }

        IdDistanceComputer prune_computer(*space_, *this);
        vamana::PruneScratch prune_scratch;
        graph_[id] = vamana::robust_prune_4bit(
            id, pool, alpha_, r_, prune_computer, &prune_scratch);
        for (vamana::NodeId neighbor : graph_[id]) {
            addBackedgeAndPrune(neighbor, id);
        }
    }

    void buildEncodedSymmetricBulk(
        const void *encoded_records,
        size_t record_count,
        size_t batch_size = 4096,
        const BuildProgressCallback &progress = BuildProgressCallback()) {
        if (encoded_records == nullptr && record_count != 0) {
            throw std::invalid_argument("Vamana bulk build received null encoded records");
        }
        if (batch_size == 0) {
            batch_size = 1;
        }
        if (record_count == 0) {
            return;
        }

        const char *records = static_cast<const char *>(encoded_records);
        prepareEmptyBulkMetadata(record_count);
        payloads_.assign(record_count * data_size_, 0);
        std::memcpy(payloads_.data(), records, record_count * data_size_);
        clearPaperPruneSidecar();
        buildGraphFromLoadedPayloads(record_count, batch_size, progress);
    }

    void buildEncodedSymmetricBulk(
        std::vector<char> &&encoded_records,
        size_t record_count,
        size_t batch_size = 4096,
        const BuildProgressCallback &progress = BuildProgressCallback()) {
        if (batch_size == 0) {
            batch_size = 1;
        }
        if (record_count == 0) {
            return;
        }
        if (encoded_records.size() != record_count * data_size_) {
            throw std::runtime_error("Vamana bulk build compact payload size mismatch");
        }
        prepareEmptyBulkMetadata(record_count);
        payloads_ = std::move(encoded_records);
        clearPaperPruneSidecar();
        buildGraphFromLoadedPayloads(record_count, batch_size, progress);
    }

    void refineGraphSymmetric(
        size_t pass_count = 1,
        size_t batch_size = 4096,
        const BuildProgressCallback &progress = BuildProgressCallback()) {
        if (pass_count == 0 || cur_element_count_ <= 1) {
            return;
        }
        if (batch_size == 0) {
            batch_size = 1;
        }
        const size_t record_count = cur_element_count_;
        const size_t total_work = pass_count * record_count;
        for (size_t pass = 0; pass < pass_count; ++pass) {
            std::vector<std::vector<vamana::NodeId>> refined(record_count);
            for (size_t begin = 0; begin < record_count; begin += batch_size) {
                const size_t end = std::min(record_count, begin + batch_size);
                std::exception_ptr first_exception;

                #pragma omp parallel
                {
                    vamana::SearchScratch search_scratch;
                    vamana::PruneScratch prune_scratch;
                    IdDistanceComputer prune_computer(*space_, *this);
                    std::vector<vamana::PruneCandidate> pool;
                    std::vector<float> existing_distances;

                    try {
                        #pragma omp for schedule(dynamic, 32)
                        for (int64_t signed_id = static_cast<int64_t>(begin);
                             signed_id < static_cast<int64_t>(end);
                             ++signed_id) {
                            const vamana::NodeId id =
                                static_cast<vamana::NodeId>(signed_id);
                            BuildEvaluator search_eval(*space_, *this, id);
                            vamana::SearchStats ignored;
                            const std::vector<vamana::SearchCandidate> &visited =
                                vamana::search_vamana(
                                    graph_, start_node_, record_count, l_build_,
                                    beam_width_, search_eval, search_scratch, &ignored);

                            pool.clear();
                            pool.reserve(visited.size() + graph_[id].size());
                            for (const vamana::SearchCandidate &candidate : visited) {
                                pool.push_back(vamana::PruneCandidate{
                                    candidate.distance, candidate.id});
                            }

                            const std::vector<vamana::NodeId> &existing = graph_[id];
                            existing_distances.resize(existing.size());
                            if (!existing.empty()) {
                                search_eval.distance_batch(
                                    existing.data(), existing.size(), existing_distances.data());
                            }
                            for (size_t i = 0; i < existing.size(); ++i) {
                                pool.push_back(vamana::PruneCandidate{
                                    existing_distances[i], existing[i]});
                            }

                            refined[id] = vamana::robust_prune_4bit(
                                id, pool, alpha_, r_, prune_computer, &prune_scratch);
                        }
                    } catch (...) {
                        #pragma omp critical
                        {
                            if (!first_exception) {
                                first_exception = std::current_exception();
                            }
                        }
                    }
                }
                if (first_exception) {
                    std::rethrow_exception(first_exception);
                }
                if (progress) {
                    progress(pass * record_count + end, total_work);
                }
            }

            graph_.swap(refined);
            std::vector<std::pair<vamana::NodeId, vamana::NodeId>> backedges;
            size_t backedge_count = 0;
            for (const auto &neighbors : graph_) {
                backedge_count += neighbors.size();
            }
            backedges.reserve(backedge_count);
            for (size_t id = 0; id < record_count; ++id) {
                for (vamana::NodeId neighbor : graph_[id]) {
                    backedges.emplace_back(neighbor, static_cast<vamana::NodeId>(id));
                }
            }
            addBackedgesGroupedAndPrune(backedges);
        }
    }

 private:
    void ensurePayloadStorage() {
        const size_t required = max_elements_ * data_size_;
        if (payloads_.size() < required) {
            payloads_.resize(required, 0);
        }
    }

    void prepareEmptyBulkMetadata(size_t record_count) {
        if (record_count > max_elements_) {
            throw std::runtime_error("Vamana bulk build exceeds index capacity");
        }
        if (cur_element_count_ != 0) {
            throw std::runtime_error("Vamana bulk build requires an empty index");
        }
        labels_.clear();
        labels_.reserve(record_count);
        label_lookup_.clear();
        graph_.assign(record_count, std::vector<vamana::NodeId>());
        for (size_t id = 0; id < record_count; ++id) {
            const labeltype label = static_cast<labeltype>(id);
            labels_.push_back(label);
            label_lookup_[label] = static_cast<vamana::NodeId>(id);
        }
        cur_element_count_ = record_count;
        start_node_ = 0;
    }

    void buildGraphFromLoadedPayloads(
        size_t record_count,
        size_t batch_size,
        const BuildProgressCallback &progress) {
        for (size_t begin = 1; begin < record_count; begin += batch_size) {
            const size_t end = std::min(record_count, begin + batch_size);
            std::vector<std::vector<vamana::NodeId>> batch_neighbors(end - begin);
            std::exception_ptr first_exception;

            #pragma omp parallel
            {
                vamana::SearchScratch search_scratch;
                vamana::PruneScratch prune_scratch;
                IdDistanceComputer prune_computer(*space_, *this);
                std::vector<vamana::PruneCandidate> pool;
                pool.reserve(l_build_);

                try {
                    #pragma omp for schedule(dynamic, 32)
                    for (int64_t signed_offset = 0;
                         signed_offset < static_cast<int64_t>(end - begin);
                         ++signed_offset) {
                        const size_t offset = static_cast<size_t>(signed_offset);
                        const vamana::NodeId id =
                            static_cast<vamana::NodeId>(begin + offset);
                        BuildEvaluator search_eval(*space_, *this, id);
                        vamana::SearchStats ignored;
                        const std::vector<vamana::SearchCandidate> &visited =
                            vamana::search_vamana(
                                graph_, start_node_, begin, l_build_, beam_width_,
                                search_eval, search_scratch, &ignored);

                        pool.clear();
                        if (pool.capacity() < visited.size()) {
                            pool.reserve(visited.size());
                        }
                        for (const vamana::SearchCandidate &candidate : visited) {
                            pool.push_back(vamana::PruneCandidate{
                                candidate.distance, candidate.id});
                        }
                        batch_neighbors[offset] = vamana::robust_prune_4bit(
                            id, pool, alpha_, r_, prune_computer, &prune_scratch);
                    }
                } catch (...) {
                    #pragma omp critical
                    {
                        if (!first_exception) {
                            first_exception = std::current_exception();
                        }
                    }
                }
            }
            if (first_exception) {
                std::rethrow_exception(first_exception);
            }

            std::vector<std::pair<vamana::NodeId, vamana::NodeId>> backedges;
            size_t backedge_count = 0;
            for (const auto &neighbors : batch_neighbors) {
                backedge_count += neighbors.size();
            }
            backedges.reserve(backedge_count);
            for (size_t offset = 0; offset < batch_neighbors.size(); ++offset) {
                const vamana::NodeId id = static_cast<vamana::NodeId>(begin + offset);
                graph_[id] = std::move(batch_neighbors[offset]);
                for (vamana::NodeId neighbor : graph_[id]) {
                    backedges.emplace_back(neighbor, id);
                }
            }
            addBackedgesGroupedAndPrune(backedges);
            if (progress) {
                progress(end, record_count);
            }
        }
    }

 public:
    std::priority_queue<std::pair<float, labeltype>>
    searchKnnPlainThenResidualRerank(
        const float *raw_query,
        size_t k,
        size_t l_search,
        size_t rerank_candidates,
        RaBitQSearchMetrics *metrics = nullptr,
        bool enable_residual_rerank = true) const {
        std::priority_queue<std::pair<float, labeltype>> result;
        if (cur_element_count_ == 0 || k == 0) {
            return result;
        }
        if (l_search < k) {
            l_search = k;
        }
        rerank_candidates = std::max(k, rerank_candidates);
        if (metrics) {
            *metrics = RaBitQSearchMetrics{};
        }
        const auto total_begin = std::chrono::steady_clock::now();
        const auto prepare_begin = std::chrono::steady_clock::now();
        QueryEvaluator evaluator(*space_, *this, raw_query);
        if (paper_prune_active_) {
            ensurePaperPruneSidecar();
        }
        const auto traversal_begin = std::chrono::steady_clock::now();
        if (metrics) {
            metrics->prepare_query_us = std::chrono::duration_cast<
                std::chrono::duration<double, std::micro>>(traversal_begin - prepare_begin).count();
        }
        vamana::SearchStats search_stats;
        std::vector<vamana::SearchCandidate> candidates =
            paper_prune_active_
                ? vamana::search_vamana_paper_active(
                    graph_, start_node_, cur_element_count_, l_search, beam_width_,
                    evaluator, paper_epsilon0_, paper_prefetch_distance_, &search_stats)
                : vamana::search_vamana(
                    graph_, start_node_, cur_element_count_, l_search, beam_width_,
                    evaluator, &search_stats);
        const auto traversal_end = std::chrono::steady_clock::now();
        if (metrics) {
            metrics->traversal_us = std::chrono::duration_cast<
                std::chrono::duration<double, std::micro>>(traversal_end - traversal_begin).count();
            metrics->visited_nodes = search_stats.visited_nodes;
            metrics->distance_computations = search_stats.distance_computations;
            metrics->hops = search_stats.hops;
            metrics->neighbors_seen = search_stats.distance_computations;
            metrics->prefetch_issued = search_stats.prefetch_issued;
            metrics->paper_checked = search_stats.paper_checked;
            metrics->paper_would_prune = search_stats.paper_would_prune;
            metrics->paper_not_pruned = search_stats.paper_not_pruned;
            metrics->paper_full_saved = search_stats.paper_full_saved;
            metrics->paper_msb_kernel_calls = search_stats.paper_msb_kernel_calls;
            metrics->paper_remaining_kernel_calls = search_stats.paper_remaining_kernel_calls;
            metrics->paper_short_time_us = search_stats.paper_short_time_us;
            metrics->paper_remaining_time_us = search_stats.paper_remaining_time_us;
        }

        const auto rerank_begin = std::chrono::steady_clock::now();
        const size_t candidate_count = std::min(rerank_candidates, candidates.size());
        std::vector<std::pair<float, vamana::NodeId>> staged;
        staged.reserve(candidate_count);
        for (size_t i = 0; i < candidate_count; ++i) {
            staged.emplace_back(candidates[i].distance, candidates[i].id);
        }

        if (enable_residual_rerank && !staged.empty()) {
            std::vector<size_t> residual_ids(staged.size(), 0);
            std::vector<const void *> residual_points(staged.size(), nullptr);
            std::vector<float> long_distances(staged.size(), 0.0f);
            std::vector<DistanceInterval> residual_intervals(staged.size());
            for (size_t i = 0; i < staged.size(); ++i) {
                residual_ids[i] = staged[i].second;
                residual_points[i] = getDataByInternalId(staged[i].second);
                long_distances[i] = staged[i].first;
            }
            space_->batch_compute_residual_distance_intervals_by_id(
                evaluator.prepared_query(),
                residual_ids.data(),
                residual_points.data(),
                long_distances.data(),
                staged.size(),
                residual_intervals.data());
            for (size_t i = 0; i < staged.size(); ++i) {
                staged[i].first = residual_intervals[i].estimate;
            }
        }
        std::sort(staged.begin(), staged.end());

        const size_t take = std::min(k, staged.size());
        for (size_t i = 0; i < take; ++i) {
            result.emplace(staged[i].first, getExternalLabel(staged[i].second));
        }
        while (result.size() > k) {
            result.pop();
        }
        const auto total_end = std::chrono::steady_clock::now();
        if (metrics) {
            metrics->rerank_us = std::chrono::duration_cast<
                std::chrono::duration<double, std::micro>>(total_end - rerank_begin).count();
            metrics->total_query_us = std::chrono::duration_cast<
                std::chrono::duration<double, std::micro>>(total_end - total_begin).count();
            metrics->active_centroids = evaluator.active_centroids();
            metrics->full_distance_count = enable_residual_rerank ? staged.size() : 0;
        }
        return result;
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(const float *raw_query, size_t k, size_t l_search, RaBitQSearchMetrics *metrics = nullptr) const {
        return searchKnnPlainThenResidualRerank(
            raw_query, k, l_search, k, metrics, false);
    }

    std::vector<std::pair<float, labeltype>>
    searchKnnCloserFirst(const float *raw_query, size_t k, size_t l_search, RaBitQSearchMetrics *metrics = nullptr) const {
        auto queue = searchKnn(raw_query, k, l_search, metrics);
        std::vector<std::pair<float, labeltype>> sorted;
        sorted.reserve(queue.size());
        while (!queue.empty()) {
            sorted.push_back(queue.top());
            queue.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        return sorted;
    }

    DegreeStats degreeStats() const {
        DegreeStats stats;
        size_t total = 0;
        for (const auto &neighbors : graph_) {
            stats.max_degree = std::max(stats.max_degree, neighbors.size());
            total += neighbors.size();
        }
        stats.avg_degree = graph_.empty()
            ? 0.0
            : static_cast<double>(total) / static_cast<double>(graph_.size());
        return stats;
    }

    size_t graphStorageBytes() const {
        size_t bytes = sizeof(uint32_t) * graph_.size();
        for (const auto &neighbors : graph_) {
            bytes += neighbors.size() * sizeof(vamana::NodeId);
        }
        bytes += labels_.size() * sizeof(labeltype);
        return bytes;
    }

    size_t payloadStorageBytes() const {
        return cur_element_count_ * data_size_;
    }

    size_t paperPruneSidecarBytes() const {
        return paper_msb_codes_.size() +
               paper_prune_factors_.size() * sizeof(PaperPruneFactors<float>);
    }

    uint64_t graphFingerprint() const {
        uint64_t hash = 1469598103934665603ULL;
        const auto mix_u64 = [&hash](uint64_t value) {
            for (size_t i = 0; i < sizeof(value); ++i) {
                hash ^= static_cast<uint8_t>(value >> (i * 8U));
                hash *= 1099511628211ULL;
            }
        };
        for (size_t i = 0; i < graph_.size(); ++i) {
            mix_u64(labels_[i]);
            for (vamana::NodeId neighbor : graph_[i]) {
                mix_u64(labels_[neighbor]);
            }
        }
        return hash;
    }

    void saveIndex(const std::string &location) const {
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("failed to open Vamana index for writing");
        }
        const char magic[8] = {'V','M','N','A','R','B','Q','1'};
        const uint32_t version = 1;
        const uint64_t max_elements = static_cast<uint64_t>(max_elements_);
        const uint64_t count = static_cast<uint64_t>(cur_element_count_);
        const uint64_t data_size = static_cast<uint64_t>(data_size_);
        const uint64_t r = static_cast<uint64_t>(r_);
        const uint64_t l_build = static_cast<uint64_t>(l_build_);
        const uint64_t beam_width = static_cast<uint64_t>(beam_width_);
        const uint64_t start_node = static_cast<uint64_t>(start_node_);
        output.write(magic, sizeof(magic));
        writeBinaryPOD(output, version);
        writeBinaryPOD(output, max_elements);
        writeBinaryPOD(output, count);
        writeBinaryPOD(output, data_size);
        writeBinaryPOD(output, r);
        writeBinaryPOD(output, l_build);
        writeBinaryPOD(output, alpha_);
        writeBinaryPOD(output, beam_width);
        writeBinaryPOD(output, start_node);
        output.write(payloads_.data(), static_cast<std::streamsize>(cur_element_count_ * data_size_));
        output.write(reinterpret_cast<const char *>(labels_.data()),
                     static_cast<std::streamsize>(labels_.size() * sizeof(labeltype)));
        for (const auto &neighbors : graph_) {
            const uint32_t degree = static_cast<uint32_t>(neighbors.size());
            writeBinaryPOD(output, degree);
            output.write(reinterpret_cast<const char *>(neighbors.data()),
                         static_cast<std::streamsize>(neighbors.size() * sizeof(vamana::NodeId)));
        }
        if (!output.good()) {
            throw std::runtime_error("failed to write Vamana index");
        }
    }

    void loadIndex(const std::string &location, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("failed to open Vamana index");
        }
        char magic[8] = {};
        uint32_t version = 0;
        uint64_t stored_max_elements = 0;
        uint64_t count = 0;
        uint64_t data_size = 0;
        uint64_t r = 0;
        uint64_t l_build = 0;
        uint64_t beam_width = 0;
        uint64_t start_node = 0;
        float alpha = 0.0f;
        input.read(magic, sizeof(magic));
        readBinaryPOD(input, version);
        readBinaryPOD(input, stored_max_elements);
        readBinaryPOD(input, count);
        readBinaryPOD(input, data_size);
        readBinaryPOD(input, r);
        readBinaryPOD(input, l_build);
        readBinaryPOD(input, alpha);
        readBinaryPOD(input, beam_width);
        readBinaryPOD(input, start_node);
        if (!input.good() || std::memcmp(magic, "VMNARBQ1", 8) != 0 || version != 1) {
            throw std::runtime_error("invalid Vamana index header");
        }
        if (data_size != space_->get_data_size()) {
            throw std::runtime_error("Vamana index payload size does not match RaBitQSpace");
        }
        max_elements_ = std::max<size_t>(
            max_elements_i, static_cast<size_t>(stored_max_elements));
        if (max_elements_ < count) {
            max_elements_ = static_cast<size_t>(count);
        }
        cur_element_count_ = static_cast<size_t>(count);
        data_size_ = static_cast<size_t>(data_size);
        r_ = static_cast<size_t>(r);
        l_build_ = static_cast<size_t>(l_build);
        alpha_ = alpha;
        beam_width_ = static_cast<size_t>(beam_width);
        start_node_ = static_cast<vamana::NodeId>(start_node);
        payloads_.assign(max_elements_ * data_size_, 0);
        labels_.assign(cur_element_count_, 0);
        graph_.assign(cur_element_count_, std::vector<vamana::NodeId>());
        label_lookup_.clear();
        clearPaperPruneSidecar();
        input.read(payloads_.data(), static_cast<std::streamsize>(cur_element_count_ * data_size_));
        input.read(reinterpret_cast<char *>(labels_.data()),
                   static_cast<std::streamsize>(labels_.size() * sizeof(labeltype)));
        for (size_t i = 0; i < cur_element_count_; ++i) {
            label_lookup_[labels_[i]] = static_cast<vamana::NodeId>(i);
        }
        for (size_t i = 0; i < cur_element_count_; ++i) {
            uint32_t degree = 0;
            readBinaryPOD(input, degree);
            graph_[i].assign(degree, 0);
            input.read(reinterpret_cast<char *>(graph_[i].data()),
                       static_cast<std::streamsize>(graph_[i].size() * sizeof(vamana::NodeId)));
        }
        if (!input.good()) {
            throw std::runtime_error("truncated Vamana index");
        }
    }

 private:
    class IdDistanceComputer {
     public:
        IdDistanceComputer(RaBitQSpace &space, const VamanaIndex &index)
            : space_(space), index_(index) {}

        IdDistanceComputer(const IdDistanceComputer &) = delete;
        IdDistanceComputer &operator=(const IdDistanceComputer &) = delete;

        ~IdDistanceComputer() {
            release_prepared();
        }

        float distance_between(vamana::NodeId lhs, vamana::NodeId rhs) {
            ensure_prepared(lhs);
            return space_.symmetric_build_distance_prepared(
                prepared_, index_.getDataByInternalId(rhs));
        }

        void distance_batch_from(
            vamana::NodeId lhs,
            const vamana::NodeId *rhs_ids,
            size_t count,
            float *distances) {
            ensure_prepared(lhs);
            points_.resize(count);
            for (size_t i = 0; i < count; ++i) {
                points_[i] = index_.getDataByInternalId(rhs_ids[i]);
            }
            space_.symmetric_build_distance_batch(
                prepared_, points_.data(), count, distances, 2);
        }

     private:
        void ensure_prepared(vamana::NodeId lhs) {
            if (prepared_ == nullptr || cached_lhs_ != lhs) {
                release_prepared();
                prepared_ = space_.prepare_symmetric_build_query(
                    index_.getDataByInternalId(lhs));
                cached_lhs_ = lhs;
            }
        }

        void release_prepared() {
            if (prepared_ != nullptr) {
                space_.release_symmetric_build_query(prepared_);
                prepared_ = nullptr;
            }
        }

        RaBitQSpace &space_;
        const VamanaIndex &index_;
        vamana::NodeId cached_lhs_{std::numeric_limits<vamana::NodeId>::max()};
        const void *prepared_{nullptr};
        std::vector<const void *> points_;
    };

    struct BackedgePruneScratch {
        std::vector<vamana::NodeId> candidates;
        std::vector<vamana::NodeId> targets;
        std::vector<float> distances;
        std::vector<vamana::PruneCandidate> pool;
    };

    void clearPaperPruneSidecar() const {
        std::lock_guard<std::mutex> lock(paper_prune_mutex_);
        paper_msb_codes_.clear();
        paper_prune_factors_.clear();
        paper_msb_stride_ = 0;
    }

    void ensurePaperPruneSidecar() const {
        if (!paper_prune_active_) {
            return;
        }
        const size_t stride = space_->paper_msb_code_bytes();
        if (stride == 0) {
            throw std::runtime_error("Vamana paper pruning sidecar is unsupported");
        }
        const size_t count = cur_element_count_;
        if (paper_msb_stride_ == stride &&
            paper_prune_factors_.size() == count &&
            paper_msb_codes_.size() == count * stride) {
            return;
        }
        std::lock_guard<std::mutex> lock(paper_prune_mutex_);
        if (paper_msb_stride_ == stride &&
            paper_prune_factors_.size() == count &&
            paper_msb_codes_.size() == count * stride) {
            return;
        }
        paper_msb_stride_ = stride;
        paper_msb_codes_.assign(count * stride, 0);
        paper_prune_factors_.resize(count);
        for (size_t id = 0; id < count; ++id) {
            paper_prune_factors_[id] = space_->extract_paper_prune_sidecar(
                getDataByInternalId(static_cast<vamana::NodeId>(id)),
                paper_msb_codes_.data() + id * stride);
        }
    }

    bool paperPruneSidecarReady() const {
        return paper_msb_stride_ != 0 &&
               paper_prune_factors_.size() == cur_element_count_ &&
               paper_msb_codes_.size() == cur_element_count_ * paper_msb_stride_;
    }

    const uint8_t *paperMsbCode(vamana::NodeId id) const {
        return paper_msb_codes_.data() + static_cast<size_t>(id) * paper_msb_stride_;
    }

    const PaperPruneFactors<float> &paperPruneFactors(vamana::NodeId id) const {
        return paper_prune_factors_[id];
    }

    void addBackedgeAndPrune(
        vamana::NodeId source,
        vamana::NodeId target,
        IdDistanceComputer *reusable_computer = nullptr,
        vamana::PruneScratch *prune_scratch = nullptr,
        BackedgePruneScratch *backedge_scratch = nullptr) {
        std::vector<vamana::NodeId> &neighbors = graph_[source];
        if (std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
            return;
        }
        neighbors.push_back(target);
        if (neighbors.size() <= r_) {
            return;
        }

        IdDistanceComputer local_computer(*space_, *this);
        IdDistanceComputer &distance_computer =
            reusable_computer ? *reusable_computer : local_computer;

        BackedgePruneScratch local_backedge_scratch;
        BackedgePruneScratch &scratch =
            backedge_scratch ? *backedge_scratch : local_backedge_scratch;

        std::vector<vamana::NodeId> &candidates = scratch.candidates;
        candidates.clear();
        candidates.reserve(neighbors.size());
        for (vamana::NodeId candidate : neighbors) {
            if (candidate != source) {
                candidates.push_back(candidate);
            }
        }
        std::vector<float> &distances = scratch.distances;
        distances.resize(candidates.size());
        if (!candidates.empty()) {
            distance_computer.distance_batch_from(
                source, candidates.data(), candidates.size(), distances.data());
        }
        std::vector<vamana::PruneCandidate> &pool = scratch.pool;
        pool.clear();
        pool.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            pool.push_back(vamana::PruneCandidate{distances[i], candidates[i]});
        }
        neighbors = vamana::robust_prune_4bit(
            source, pool, alpha_, r_, distance_computer, prune_scratch);
    }

    void addBackedgesGroupedAndPrune(
        std::vector<std::pair<vamana::NodeId, vamana::NodeId>> &backedges) {
        if (backedges.empty()) {
            return;
        }
        std::sort(backedges.begin(), backedges.end());

        std::vector<size_t> group_offsets;
        group_offsets.reserve(backedges.size() + 1U);
        group_offsets.push_back(0);
        for (size_t i = 1; i < backedges.size(); ++i) {
            if (backedges[i].first != backedges[i - 1].first) {
                group_offsets.push_back(i);
            }
        }
        group_offsets.push_back(backedges.size());

        std::exception_ptr first_exception;
        #pragma omp parallel
        {
            IdDistanceComputer distance_computer(*space_, *this);
            vamana::PruneScratch prune_scratch;
            BackedgePruneScratch backedge_scratch;

            try {
                #pragma omp for schedule(dynamic, 32)
                for (int64_t signed_group = 0;
                     signed_group < static_cast<int64_t>(group_offsets.size() - 1U);
                     ++signed_group) {
                    const size_t group = static_cast<size_t>(signed_group);
                    addBackedgeGroupAndPrune(
                        backedges,
                        group_offsets[group],
                        group_offsets[group + 1U],
                        &distance_computer,
                        &prune_scratch,
                        &backedge_scratch);
                }
            } catch (...) {
                #pragma omp critical
                {
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
            }
        }
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

    void addBackedgeGroupAndPrune(
        const std::vector<std::pair<vamana::NodeId, vamana::NodeId>> &backedges,
        size_t begin,
        size_t end,
        IdDistanceComputer *reusable_computer,
        vamana::PruneScratch *prune_scratch,
        BackedgePruneScratch *backedge_scratch) {
        if (begin >= end) {
            return;
        }
        const vamana::NodeId source = backedges[begin].first;
        std::vector<vamana::NodeId> &neighbors = graph_[source];

        BackedgePruneScratch local_backedge_scratch;
        BackedgePruneScratch &scratch =
            backedge_scratch ? *backedge_scratch : local_backedge_scratch;

        std::vector<vamana::NodeId> &targets = scratch.targets;
        targets.clear();
        targets.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            if (backedges[i].second != source) {
                targets.push_back(backedges[i].second);
            }
        }
        if (targets.empty()) {
            return;
        }
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

        size_t added = 0;
        for (vamana::NodeId target : targets) {
            if (std::find(neighbors.begin(), neighbors.end(), target) == neighbors.end()) {
                neighbors.push_back(target);
                ++added;
            }
        }
        if (added == 0 || neighbors.size() <= r_) {
            return;
        }

        IdDistanceComputer local_computer(*space_, *this);
        IdDistanceComputer &distance_computer =
            reusable_computer ? *reusable_computer : local_computer;

        std::vector<vamana::NodeId> &candidates = scratch.candidates;
        candidates.clear();
        candidates.reserve(neighbors.size());
        for (vamana::NodeId candidate : neighbors) {
            if (candidate != source) {
                candidates.push_back(candidate);
            }
        }

        std::vector<float> &distances = scratch.distances;
        distances.resize(candidates.size());
        if (!candidates.empty()) {
            distance_computer.distance_batch_from(
                source, candidates.data(), candidates.size(), distances.data());
        }

        std::vector<vamana::PruneCandidate> &pool = scratch.pool;
        pool.clear();
        pool.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            pool.push_back(vamana::PruneCandidate{distances[i], candidates[i]});
        }
        neighbors = vamana::robust_prune_4bit(
            source, pool, alpha_, r_, distance_computer, prune_scratch);
    }

    RaBitQSpace *space_{nullptr};
    size_t max_elements_{0};
    size_t cur_element_count_{0};
    size_t data_size_{0};
    size_t r_{kDefaultR};
    size_t l_build_{kDefaultLBuild};
    float alpha_{kDefaultAlpha};
    size_t beam_width_{kDefaultBeamWidth};
    bool paper_prune_active_{true};
    float paper_epsilon0_{1.9f};
    size_t paper_prefetch_distance_{2};
    vamana::NodeId start_node_{0};
    std::vector<char> payloads_;
    std::vector<labeltype> labels_;
    std::unordered_map<labeltype, vamana::NodeId> label_lookup_;
    std::vector<std::vector<vamana::NodeId>> graph_;
    mutable std::mutex paper_prune_mutex_;
    mutable std::vector<uint8_t> paper_msb_codes_;
    mutable std::vector<PaperPruneFactors<float>> paper_prune_factors_;
    mutable size_t paper_msb_stride_{0};
};

}  // namespace hnswlib
