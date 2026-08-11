#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
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
    struct RouteSidecarHeader {
        char magic[8];
        uint32_t version;
        uint32_t route_bits;
        uint32_t strategy;
        uint32_t zero_sign_rule;
        uint64_t element_count;
        uint64_t graph_fingerprint;
        uint64_t route_fingerprint;
        uint64_t dimension_count;
    };

    static std::string routeStatePath(const std::string &location) { return location + ".route"; }

    static uint64_t routeFingerprint(
        const std::vector<uint32_t> &dims,
        const std::vector<uint32_t> &codes) {
        uint64_t hash = 1469598103934665603ULL;
        const auto mix = [&hash](uint32_t value) {
            for (size_t i = 0; i < sizeof(value); ++i) {
                hash ^= static_cast<uint8_t>(value >> (i * 8));
                hash *= 1099511628211ULL;
            }
        };
        for (uint32_t dim : dims) mix(dim);
        for (uint32_t code : codes) mix(code);
        return hash;
    }

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
        GraphTurboConfig config;
        config.mode = GraphTurboMode::BatchPrefetch;
        config.prefetch_distance = 2;
        index_.setGraphTurboConfig(config);
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
        GraphTurboConfig config;
        config.mode = GraphTurboMode::BatchPrefetch;
        config.prefetch_distance = 2;
        index_.setGraphTurboConfig(config);
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

    void setGraphTurboConfig(const GraphTurboConfig &config) { index_.setGraphTurboConfig(config); }
    const GraphTurboConfig &getGraphTurboConfig() const { return index_.getGraphTurboConfig(); }
    void freezeGraphTurboTopology(bool frozen = true) { index_.freezeGraphTurboTopology(frozen); }
    bool graphTurboTopologyFrozen() const { return index_.graphTurboTopologyFrozen(); }
    void setRouteOracle(const float *base_vectors_by_internal_id, size_t count) {
        index_.setRouteOracle(base_vectors_by_internal_id, count, space_.get_dim());
    }

    void buildRouteCodes(
        const float *base_vectors_by_internal_id,
        size_t count,
        RouteCodeStrategy strategy,
        uint32_t route_bits) {
        if (base_vectors_by_internal_id == nullptr || count != index_.cur_element_count)
            throw std::invalid_argument("route base vectors must match internal-id count");
        if (space_.get_centroid_count() != 1 || route_bits < 1 || route_bits > 32 ||
            route_bits > space_.get_code_dim())
            throw std::invalid_argument("route codes require K=1 and valid route_bits");
        std::vector<uint32_t> dims(route_bits);
        const size_t code_dim = space_.get_code_dim();
        if (strategy == RouteCodeStrategy::HighVariance) {
            std::vector<double> sum(code_dim, 0.0), sum_squares(code_dim, 0.0);
            std::vector<float> residual;
            for (size_t id = 0; id < count; ++id) {
                space_.compute_rotated_residual_k1(
                    base_vectors_by_internal_id + id * space_.get_dim(), residual);
                for (size_t d = 0; d < code_dim; ++d) {
                    sum[d] += residual[d];
                    sum_squares[d] += static_cast<double>(residual[d]) * residual[d];
                }
            }
            std::vector<std::pair<double, uint32_t>> ranked;
            ranked.reserve(code_dim);
            for (size_t d = 0; d < code_dim; ++d) {
                const double mean = sum[d] / static_cast<double>(count);
                ranked.emplace_back(sum_squares[d] / static_cast<double>(count) - mean * mean,
                                    static_cast<uint32_t>(d));
            }
            std::partial_sort(ranked.begin(), ranked.begin() + route_bits, ranked.end(),
                [](const std::pair<double, uint32_t> &a,
                   const std::pair<double, uint32_t> &b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second;
                });
            for (size_t i = 0; i < route_bits; ++i) dims[i] = ranked[i].second;
        } else {
            for (size_t i = 0; i < route_bits; ++i) {
                dims[i] = strategy == RouteCodeStrategy::EqualInterval
                    ? static_cast<uint32_t>(((2 * i + 1) * code_dim) / (2 * route_bits))
                    : static_cast<uint32_t>((i * code_dim) / route_bits);
            }
        }
        std::vector<uint32_t> codes(count);
        for (size_t id = 0; id < count; ++id) {
            codes[id] = space_.compute_database_route_code(
                base_vectors_by_internal_id + id * space_.get_dim(), dims);
        }
        index_.setRouteCodeStorage(
            std::make_shared<ContiguousRouteCodeStorage>(std::move(codes)), std::move(dims));
        GraphTurboConfig config = index_.getGraphTurboConfig();
        config.route_strategy = strategy;
        config.route_bits = route_bits;
        index_.setGraphTurboConfig(config);
        index_.freezeGraphTurboTopology(true);
    }

    uint64_t routeCodeFingerprint() const {
        const auto storage = std::dynamic_pointer_cast<const ContiguousRouteCodeStorage>(
            index_.routeCodeStorage());
        if (!storage) return 0;
        return routeFingerprint(index_.routeDimensions(), storage->codes());
    }

    void saveRouteCodes(const std::string &location) const {
        const auto storage = std::dynamic_pointer_cast<const ContiguousRouteCodeStorage>(
            index_.routeCodeStorage());
        if (!storage) {
            std::remove(routeStatePath(location).c_str());
            return;
        }
        std::ofstream output(routeStatePath(location), std::ios::binary);
        if (!output) throw std::runtime_error("failed to open Graph-Turbo route sidecar");
        RouteSidecarHeader header{};
        std::memcpy(header.magic, "GTRTQ01", 8);
        header.version = 1;
        header.route_bits = static_cast<uint32_t>(index_.routeDimensions().size());
        header.strategy = static_cast<uint32_t>(index_.getGraphTurboConfig().route_strategy);
        header.zero_sign_rule = 1;
        header.element_count = index_.cur_element_count;
        header.graph_fingerprint = index_.graphFingerprint();
        header.route_fingerprint = routeFingerprint(index_.routeDimensions(), storage->codes());
        header.dimension_count = index_.routeDimensions().size();
        output.write(reinterpret_cast<const char *>(&header), sizeof(header));
        output.write(reinterpret_cast<const char *>(index_.routeDimensions().data()),
                     index_.routeDimensions().size() * sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(storage->codes().data()),
                     storage->codes().size() * sizeof(uint32_t));
        if (!output.good()) throw std::runtime_error("failed to write Graph-Turbo route sidecar");
    }

    bool loadRouteCodes(const std::string &location, bool required = false) {
        std::ifstream input(routeStatePath(location), std::ios::binary);
        if (!input) {
            if (required) throw std::runtime_error("missing Graph-Turbo route sidecar");
            index_.clearRouteCodeStorage();
            index_.freezeGraphTurboTopology(false);
            return false;
        }
        RouteSidecarHeader header{};
        input.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (!input.good() || std::memcmp(header.magic, "GTRTQ01", 8) != 0 ||
            header.version != 1 || header.zero_sign_rule != 1 ||
            header.element_count != index_.cur_element_count ||
            header.graph_fingerprint != index_.graphFingerprint() ||
            header.route_bits < 1 || header.route_bits > 32 ||
            header.strategy > static_cast<uint32_t>(RouteCodeStrategy::ShortCodeSelected) ||
            header.dimension_count != header.route_bits)
            throw std::runtime_error("invalid or mismatched Graph-Turbo route sidecar");
        std::vector<uint32_t> dims(header.dimension_count), codes(header.element_count);
        input.read(reinterpret_cast<char *>(dims.data()), dims.size() * sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(codes.data()), codes.size() * sizeof(uint32_t));
        if (!input.good() || input.peek() != std::ifstream::traits_type::eof() ||
            routeFingerprint(dims, codes) != header.route_fingerprint)
            throw std::runtime_error("corrupt Graph-Turbo route sidecar");
        index_.setRouteCodeStorage(
            std::make_shared<ContiguousRouteCodeStorage>(std::move(codes)), std::move(dims));
        GraphTurboConfig config = index_.getGraphTurboConfig();
        config.route_bits = header.route_bits;
        config.route_strategy = static_cast<RouteCodeStrategy>(header.strategy);
        index_.setGraphTurboConfig(config);
        index_.freezeGraphTurboTopology(true);
        return true;
    }

    uint64_t graphFingerprint() const {
        return index_.graphFingerprint();
    }

    uint64_t labelGraphFingerprint() const { return index_.labelGraphFingerprint(); }
    double averageLevel0NeighborIdDistance() const {
        return index_.averageLevel0NeighborIdDistance();
    }

    uint64_t payloadFingerprintByLabel() const {
        uint64_t hash = 1469598103934665603ULL;
        std::vector<std::pair<labeltype, tableint>> nodes;
        nodes.reserve(index_.cur_element_count);
        for (tableint id = 0; id < index_.cur_element_count; ++id)
            nodes.emplace_back(index_.getExternalLabel(id), id);
        std::sort(nodes.begin(), nodes.end());
        for (const auto &node : nodes) {
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(
                index_.getDataByInternalId(node.second));
            for (size_t i = 0; i < index_.data_size_; ++i) {
                hash ^= bytes[i];
                hash *= 1099511628211ULL;
            }
        }
        return hash;
    }

    uint64_t residualFingerprintByLabel() const {
        if (!space_.external_residual_storage_enabled()) return 0;
        uint64_t hash = 1469598103934665603ULL;
        std::vector<std::pair<labeltype, tableint>> nodes;
        nodes.reserve(index_.cur_element_count);
        for (tableint id = 0; id < index_.cur_element_count; ++id)
            nodes.emplace_back(index_.getExternalLabel(id), id);
        std::sort(nodes.begin(), nodes.end());
        std::vector<char> record(space_.get_residual_disk_record_bytes());
        for (const auto &node : nodes) {
            space_.copyExternalResidualRecord(node.second, record.data());
            for (uint8_t byte : record) {
                hash ^= byte;
                hash *= 1099511628211ULL;
            }
        }
        return hash;
    }

    void importBfsReorderedFrom(
        const RaBitQHierarchicalNSW &source,
        const std::string &target_residual_path = std::string()) {
        if (index_.max_elements_ < source.index_.cur_element_count)
            throw std::runtime_error("BFS reorder target capacity is too small");
        std::stringstream quantizer_state(std::ios::in | std::ios::out | std::ios::binary);
        source.space_.saveState(quantizer_state);
        quantizer_state.seekg(0);
        space_.loadState(quantizer_state);
        const std::vector<tableint> new_to_old = source.index_.bfsOrderLevel0();
        index_.importGraphAndCopyDataFromReordered(
            source.index_, new_to_old,
            [&source](tableint old_id, tableint, void *target_data) {
                std::memcpy(target_data, source.index_.getDataByInternalId(old_id),
                            source.index_.data_size_);
            });
        GraphTurboConfig config = source.index_.getGraphTurboConfig();
        config.mode = GraphTurboMode::BatchPrefetch;
        config.prefetch_distance = 2;
        index_.setGraphTurboConfig(config);
        if (source.space_.external_residual_storage_enabled()) {
            if (target_residual_path.empty())
                throw std::invalid_argument("BFS reorder requires a target residual path");
            const size_t record_size = source.space_.get_residual_disk_record_bytes();
            const int fd = ::open(target_residual_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
            if (fd < 0) throw std::runtime_error("cannot create BFS residual sidecar");
            const size_t total_bytes = new_to_old.size() * record_size;
            if (::ftruncate(fd, static_cast<off_t>(total_bytes)) != 0) {
                ::close(fd);
                throw std::runtime_error("cannot resize BFS residual sidecar");
            }
            std::vector<char> record(record_size);
            for (tableint new_id = 0; new_id < new_to_old.size(); ++new_id) {
                source.space_.copyExternalResidualRecord(new_to_old[new_id], record.data());
                size_t written = 0;
                while (written < record_size) {
                    const ssize_t rc = ::pwrite(fd, record.data() + written, record_size - written,
                        static_cast<off_t>(static_cast<size_t>(new_id) * record_size + written));
                    if (rc <= 0) {
                        ::close(fd);
                        throw std::runtime_error("failed writing BFS residual sidecar");
                    }
                    written += static_cast<size_t>(rc);
                }
            }
            ::close(fd);
            space_.openExternalResidualStorage(target_residual_path, new_to_old.size());
        }
        if (labelGraphFingerprint() != source.labelGraphFingerprint() ||
            payloadFingerprintByLabel() != source.payloadFingerprintByLabel() ||
            residualFingerprintByLabel() != source.residualFingerprintByLabel())
            throw std::runtime_error("BFS reorder integrity verification failed");
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
        saveRouteCodes(location);
    }
//加载索引
    void loadIndex(const std::string &location, size_t max_elements = 0) {
        index_.setRouteOracle(nullptr, 0, 0);
        std::ifstream quantizer_input(quantizerStatePath(location), std::ios::binary);
        if (!quantizer_input.is_open()) {
            throw std::runtime_error(
                "RaBitQHierarchicalNSW missing quantizer state file; expected " + quantizerStatePath(location));
        }
        space_.loadState(quantizer_input);
        quantizer_input.close();
        index_.loadIndex(location, &space_, max_elements);
        loadRouteCodes(location, false);
        space_.openExternalResidualStorage(residualPathForIndex(location), index_.cur_element_count);
    }
//添加数据点，首先将原始向量编码成RaBitQ编码，然后调用index_的addPoint方法插入编码后的向量
    void addPoint(const float *raw_vector, labeltype label, bool replace_deleted = false) {
        std::vector<char> encoded = space_.encodeVector(raw_vector);
        index_.addPoint(encoded.data(), label, replace_deleted);
    }

    void setAsymmetricBuildRawProvider(
        std::function<const void *(labeltype)> provider) {
        index_.setAsymmetricBuildRawProvider(std::move(provider));
    }

    void clearAsymmetricBuildRawProvider() {
        index_.clearAsymmetricBuildRawProvider();
    }

    void addPointAsymmetric(
        const float *raw_vector,
        labeltype label,
        const void *compact_encoded = nullptr) {
        if (space_.get_centroid_count() != 1)
            throw std::runtime_error(
                "asymmetric 4-bit construction requires K=1");
        if (compact_encoded != nullptr) {
            index_.addPointAsymmetric(compact_encoded, raw_vector, label);
            return;
        }
        std::vector<char> encoded = space_.encodeVector(raw_vector);
        index_.addPointAsymmetric(encoded.data(), raw_vector, label);
    }

    uint64_t asymmetricBuildDistanceCalls() const {
        return index_.asymmetricBuildDistanceCalls();
    }

    uint64_t encodedBuildDistanceCalls() const {
        return index_.encodedBuildDistanceCalls();
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

    void materializeExternalResidualsFromFullPayloadFile(
        const std::string &payload_path,
        const std::string &residual_path,
        size_t full_record_size) {
        if (!space_.external_residual_storage_enabled())
            throw std::runtime_error("RaBitQ external residual storage is disabled");
        if (full_record_size != space_.get_full_data_size())
            throw std::runtime_error("RaBitQ full payload record size mismatch");
        std::ifstream payload_input(payload_path, std::ios::binary);
        if (!payload_input.is_open())
            throw std::runtime_error("RaBitQ failed to open full payload file: " + payload_path);
        const size_t count = index_.cur_element_count;
        const size_t residual_size = space_.get_residual_disk_record_bytes();
        const int output_fd = ::open(residual_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
        if (output_fd < 0)
            throw std::runtime_error("RaBitQ failed to create residual file: " + residual_path);
        if (::ftruncate(output_fd, static_cast<off_t>(count * residual_size)) != 0) {
            ::close(output_fd);
            throw std::runtime_error("RaBitQ failed to resize residual file: " + residual_path);
        }
        std::vector<char> full(full_record_size);
        std::vector<char> residual(residual_size);
        try {
            for (tableint internal_id = 0; internal_id < count; ++internal_id) {
                const labeltype label = index_.getExternalLabel(internal_id);
                if (label >= count)
                    throw std::runtime_error("RaBitQ external label is outside payload file");
                payload_input.clear();
                payload_input.seekg(static_cast<std::streamoff>(label * full_record_size));
                payload_input.read(full.data(), static_cast<std::streamsize>(full_record_size));
                if (!payload_input.good())
                    throw std::runtime_error("RaBitQ failed reading full payload record");
                space_.copyResidualRecordFromFull(full.data(), residual.data());
                size_t written = 0;
                while (written < residual_size) {
                    const ssize_t rc = ::pwrite(
                        output_fd, residual.data() + written, residual_size - written,
                        static_cast<off_t>(internal_id * residual_size + written));
                    if (rc < 0 && errno == EINTR) continue;
                    if (rc <= 0) throw std::runtime_error("RaBitQ failed writing residual record");
                    written += static_cast<size_t>(rc);
                }
            }
        } catch (...) {
            ::close(output_fd);
            throw;
        }
        ::close(output_fd);
        space_.openExternalResidualStorage(residual_path, count);
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
        BaseFilterFunctor *isIdAllowed = nullptr,
        RaBitQSearchMetrics *metrics = nullptr) const {
        return index_.searchKnnPlainThenResidualRerank(
            raw_query, k, rerank_candidates, isIdAllowed, metrics);
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnnPrimaryOnly(
        const float *raw_query,
        size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        RaBitQSearchMetrics *metrics = nullptr) const {
        return index_.searchKnnPlainThenResidualRerank(
            raw_query, k, k, isIdAllowed, metrics, false);
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
