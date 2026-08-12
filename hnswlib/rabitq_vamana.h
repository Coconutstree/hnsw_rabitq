#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "space_rabitq.h"
#include "vamana_index.h"

namespace hnswlib {

class RaBitQVamanaIndex {
 private:
    static std::string quantizerStatePath(const std::string &location) {
        return location + ".rabitq";
    }

    static std::string residualPathForIndex(const std::string &location) {
        return location + ".residual";
    }

    RaBitQSpace space_;
    VamanaIndex index_;
    size_t l_search_{100};

 public:
    RaBitQVamanaIndex(
        size_t dim,
        size_t max_elements,
        size_t centroid_count = 1,
        size_t random_seed = 100,
        bool external_residual_storage = false,
        RaBitQSpace::ResidualQuantizationConfig residual_config = RaBitQSpace::ResidualQuantizationConfig())
        : space_(dim,
                 centroid_count,
                 static_cast<uint32_t>(random_seed),
                 external_residual_storage,
                 residual_config),
          index_(&space_,
                 max_elements,
                 VamanaIndex::kDefaultR,
                 VamanaIndex::kDefaultLBuild,
                 VamanaIndex::kDefaultAlpha,
                 VamanaIndex::kDefaultBeamWidth) {
        if (centroid_count != 1) {
            throw std::runtime_error("Vamana ExRaBitQ symmetric build currently requires K=1");
        }
    }

    RaBitQSpace &space() { return space_; }
    const RaBitQSpace &space() const { return space_; }
    VamanaIndex &index() { return index_; }
    const VamanaIndex &index() const { return index_; }

    void setLSearch(size_t l_search) {
        if (l_search == 0) {
            throw std::invalid_argument("L_search must be positive");
        }
        l_search_ = l_search;
    }

    size_t getLSearch() const { return l_search_; }

    void setPaperPrune(bool active, float epsilon0 = 1.9f) {
        index_.setPaperPrune(active, epsilon0);
    }

    bool paperPruneActive() const { return index_.paper_prune_active(); }
    float paperEpsilon0() const { return index_.paper_epsilon0(); }

    void preparePaperPruneSidecar() const {
        index_.preparePaperPruneSidecar();
    }

    void addPoint(const float *raw_vector, labeltype label) {
        std::vector<char> encoded = space_.encodeVector(raw_vector);
        index_.addPointEncodedSymmetric(encoded.data(), label);
    }

    void addPointEncodedSymmetric(const void *encoded, labeltype label) {
        index_.addPointEncodedSymmetric(encoded, label);
    }

    void buildEncodedSymmetricBulk(
        const void *encoded_records,
        size_t record_count,
        size_t batch_size = 4096) {
        index_.buildEncodedSymmetricBulk(encoded_records, record_count, batch_size);
    }

    void buildEncodedSymmetricBulk(
        std::vector<char> &&encoded_records,
        size_t record_count,
        size_t batch_size,
        const VamanaIndex::BuildProgressCallback &progress) {
        index_.buildEncodedSymmetricBulk(
            std::move(encoded_records), record_count, batch_size, progress);
    }

    void refineGraphSymmetric(
        size_t pass_count,
        size_t batch_size,
        const VamanaIndex::BuildProgressCallback &progress =
            VamanaIndex::BuildProgressCallback()) {
        index_.refineGraphSymmetric(pass_count, batch_size, progress);
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnnPrimaryOnly(
        const float *raw_query,
        size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        RaBitQSearchMetrics *metrics = nullptr) const {
        if (isIdAllowed != nullptr) {
            throw std::runtime_error("Vamana path does not support filtered search");
        }
        return index_.searchKnn(raw_query, k, l_search_, metrics);
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnnPlainThenResidualRerank(
        const float *raw_query,
        size_t k,
        size_t rerank_candidates,
        BaseFilterFunctor *isIdAllowed = nullptr,
        RaBitQSearchMetrics *metrics = nullptr) const {
        if (isIdAllowed != nullptr) {
            throw std::runtime_error("Vamana path does not support filtered search");
        }
        return index_.searchKnnPlainThenResidualRerank(
            raw_query, k, l_search_, rerank_candidates, metrics, true);
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(
        const float *raw_query,
        size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        return searchKnnPrimaryOnly(raw_query, k, isIdAllowed, nullptr);
    }

    std::vector<std::pair<float, labeltype>>
    searchKnnPrimaryOnlyCloserFirst(
        const float *raw_query,
        size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        RaBitQSearchMetrics *metrics = nullptr) const {
        if (isIdAllowed != nullptr) {
            throw std::runtime_error("Vamana path does not support filtered search");
        }
        return index_.searchKnnCloserFirst(raw_query, k, l_search_, metrics);
    }

    std::vector<std::pair<float, labeltype>>
    searchKnnPlainThenResidualRerankCloserFirst(
        const float *raw_query,
        size_t k,
        size_t rerank_candidates,
        BaseFilterFunctor *isIdAllowed = nullptr,
        RaBitQSearchMetrics *metrics = nullptr) const {
        auto result = searchKnnPlainThenResidualRerank(
            raw_query, k, rerank_candidates, isIdAllowed, metrics);
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
    searchKnnCloserFirst(
        const float *raw_query,
        size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        return searchKnnPrimaryOnlyCloserFirst(raw_query, k, isIdAllowed, nullptr);
    }

    void saveIndex(const std::string &location) const {
        std::ofstream quantizer_output(quantizerStatePath(location), std::ios::binary);
        if (!quantizer_output.is_open()) {
            throw std::runtime_error("RaBitQVamanaIndex failed to open quantizer state file");
        }
        space_.saveState(quantizer_output);
        quantizer_output.close();
        index_.saveIndex(location);
    }

    void loadIndex(const std::string &location, size_t max_elements = 0) {
        std::ifstream quantizer_input(quantizerStatePath(location), std::ios::binary);
        if (!quantizer_input.is_open()) {
            throw std::runtime_error(
                "RaBitQVamanaIndex missing quantizer state file: " + quantizerStatePath(location));
        }
        space_.loadState(quantizer_input);
        quantizer_input.close();
        index_.loadIndex(location, max_elements);
        std::ifstream residual_input(residualPathForIndex(location), std::ios::binary);
        if (residual_input.good()) {
            residual_input.close();
            space_.openExternalResidualStorage(residualPathForIndex(location), index_.size());
        }
    }

    void materializeExternalResidualsFromFullPayloadFile(
        const std::string &payload_path,
        const std::string &residual_path,
        size_t full_record_size) {
        if (!space_.external_residual_storage_enabled()) {
            throw std::runtime_error("RaBitQ external residual storage is disabled");
        }
        if (full_record_size != space_.get_full_data_size()) {
            throw std::runtime_error("RaBitQ full payload record size mismatch");
        }
        std::ifstream payload_input(payload_path, std::ios::binary);
        if (!payload_input.is_open()) {
            throw std::runtime_error("RaBitQ failed to open full payload file: " + payload_path);
        }
        const size_t residual_size = space_.get_residual_disk_record_bytes();
        const int output_fd = ::open(residual_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
        if (output_fd < 0) {
            throw std::runtime_error("RaBitQ failed to create residual file: " + residual_path);
        }
        if (::ftruncate(output_fd, static_cast<off_t>(index_.size() * residual_size)) != 0) {
            ::close(output_fd);
            throw std::runtime_error("RaBitQ failed to resize residual file: " + residual_path);
        }
        std::vector<char> full(full_record_size, 0);
        std::vector<char> residual(residual_size, 0);
        try {
            for (size_t internal_id = 0; internal_id < index_.size(); ++internal_id) {
                const labeltype label = index_.getExternalLabel(static_cast<vamana::NodeId>(internal_id));
                payload_input.clear();
                payload_input.seekg(static_cast<std::streamoff>(label * full_record_size));
                payload_input.read(full.data(), static_cast<std::streamsize>(full_record_size));
                if (!payload_input.good()) {
                    throw std::runtime_error("RaBitQ failed reading full payload record");
                }
                space_.copyResidualRecordFromFull(full.data(), residual.data());
                size_t written = 0;
                while (written < residual_size) {
                    const ssize_t rc = ::pwrite(
                        output_fd,
                        residual.data() + written,
                        residual_size - written,
                        static_cast<off_t>(internal_id * residual_size + written));
                    if (rc < 0 && errno == EINTR) {
                        continue;
                    }
                    if (rc <= 0) {
                        throw std::runtime_error("RaBitQ failed writing residual record");
                    }
                    written += static_cast<size_t>(rc);
                }
            }
        } catch (...) {
            ::close(output_fd);
            throw;
        }
        ::close(output_fd);
        space_.openExternalResidualStorage(residual_path, index_.size());
    }

    uint64_t graphFingerprint() const { return index_.graphFingerprint(); }
    size_t graphStorageBytes() const { return index_.graphStorageBytes(); }
    size_t payloadStorageBytes() const { return index_.payloadStorageBytes(); }
    size_t paperPruneSidecarBytes() const { return index_.paperPruneSidecarBytes(); }
    VamanaIndex::DegreeStats degreeStats() const { return index_.degreeStats(); }
};

}  // namespace hnswlib
