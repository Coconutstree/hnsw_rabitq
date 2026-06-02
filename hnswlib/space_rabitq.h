#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <istream>
#include <ostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hnswlib.h"

namespace hnswlib {

class RaBitQSpace : public SpaceInterface<float> {
 public:
    static constexpr size_t kBaseBits = 4;
    static constexpr uint8_t kBaseMax = (1U << kBaseBits) - 1U;
    static constexpr int kQueryBits = 6;
    static constexpr size_t kFactorCount = 3;

    struct EncodedHeader {
        float norm_sqr;
        float factor_dq;
        float factor_vq;
    };

    struct QueryContext {
        std::vector<uint8_t> byte_query;
        float query_norm = 0.0f;
        float query_norm_sqr = 0.0f;
        float lower_val = 0.0f;
        float width = 1.0f;
        int32_t sumq = 0;
    };

 private:
    size_t dim_{0};
    size_t code_dim_{0};
    size_t compact_code_bytes_{0};
    size_t data_size_{0};
    float inv_sqrt_code_dim_{1.0f};

    DISTFUNC<float> fstdistfunc_{nullptr};

    std::vector<float> projection_;
    std::vector<float> global_center_;
    std::vector<float> rotated_global_center_;

    static size_t roundUp64(size_t value) {
        return ((value + 63U) / 64U) * 64U;
    }

    static float exrabitqDistance(const void *lhs, const void *rhs, const void *space_ptr) {
        const RaBitQSpace *space = static_cast<const RaBitQSpace *>(space_ptr);
        return space->distanceBetweenEncoded(
            static_cast<const char *>(lhs),
            static_cast<const char *>(rhs));
    }

    EncodedHeader loadHeader(const void *encoded) const {
        EncodedHeader value;
        std::memcpy(&value, encoded, sizeof(EncodedHeader));
        return value;
    }

    const uint8_t *codeBytes(const void *encoded) const {
        return reinterpret_cast<const uint8_t *>(static_cast<const char *>(encoded) + sizeof(EncodedHeader));
    }

    uint8_t *codeBytes(void *encoded) const {
        return reinterpret_cast<uint8_t *>(static_cast<char *>(encoded) + sizeof(EncodedHeader));
    }

    void rotate(const float *raw_vector, std::vector<float> &rotated) const {
        rotated.assign(code_dim_, 0.0f);
        for (size_t row = 0; row < code_dim_; ++row) {
            const float *proj_row = projection_.data() + row * code_dim_;
            float value = 0.0f;
            for (size_t col = 0; col < dim_; ++col) {
                value += proj_row[col] * raw_vector[col];
            }
            rotated[row] = value;
        }
    }

    void refreshRotatedCenter() {
        rotate(global_center_.data(), rotated_global_center_);
    }

    void exrabitqGlobalCode(
        const float *rotated_unit_data,
        float residual_norm,
        uint8_t *compact_code,
        EncodedHeader &header) const {
        constexpr double eps = 1e-5;
        constexpr int n_enum = 10;

        double max_o = -1.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            max_o = std::max(max_o, static_cast<double>(rotated_unit_data[i]));
        }

        double t_start = static_cast<double>(kBaseMax / 3U) / (max_o + eps);
        double t_end = static_cast<double>(kBaseMax + n_enum) / (max_o + eps);

        std::vector<int> cur_o_bar(code_dim_, 0);
        double sqr_denominator = static_cast<double>(code_dim_) * 0.25;
        double numerator = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            cur_o_bar[i] = static_cast<int>(t_start * rotated_unit_data[i] + eps);
            cur_o_bar[i] = std::min<int>(kBaseMax, std::max(0, cur_o_bar[i]));
            sqr_denominator += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
            numerator += (cur_o_bar[i] + 0.5) * rotated_unit_data[i];
        }

        std::priority_queue<
            std::pair<double, size_t>,
            std::vector<std::pair<double, size_t>>,
            std::greater<std::pair<double, size_t>>> next_t;

        for (size_t i = 0; i < code_dim_; ++i) {
            if (rotated_unit_data[i] > eps) {
                next_t.emplace((cur_o_bar[i] + 1) / (rotated_unit_data[i] + eps), i);
            }
        }

        double max_ip = 0.0;
        double best_t = t_start;
        while (!next_t.empty()) {
            const double cur_t = next_t.top().first;
            const size_t update_id = next_t.top().second;
            next_t.pop();

            if (cur_t > t_end) {
                break;
            }

            cur_o_bar[update_id]++;
            const int update_o_bar = cur_o_bar[update_id];
            sqr_denominator += 2 * update_o_bar;
            numerator += rotated_unit_data[update_id];

            const double cur_ip = numerator / std::sqrt(sqr_denominator);
            if (cur_ip > max_ip) {
                max_ip = cur_ip;
                best_t = cur_t;
            }

            if (update_o_bar < static_cast<int>(kBaseMax)) {
                const double t_next = (update_o_bar + 1) / (rotated_unit_data[update_id] + eps);
                if (t_next < t_end) {
                    next_t.emplace(t_next, update_id);
                }
            }
        }

        std::vector<int> o_bar(code_dim_, 0);
        int code_sum = 0;
        float x0 = 0.0f;
        std::memset(compact_code, 0, compact_code_bytes_);

        for (size_t i = 0; i < code_dim_; ++i) {
            o_bar[i] = static_cast<int>(best_t * rotated_unit_data[i] + eps);
            o_bar[i] = std::min<int>(kBaseMax, std::max(0, o_bar[i]));
            code_sum += o_bar[i];
            x0 += rotated_unit_data[i] * static_cast<float>(o_bar[i]) * inv_sqrt_code_dim_;

            if ((i & 1U) == 0) {
                compact_code[i / 2] = static_cast<uint8_t>(o_bar[i] & 0x0F);
            } else {
                compact_code[i / 2] |= static_cast<uint8_t>((o_bar[i] & 0x0F) << 4);
            }
        }

        header.norm_sqr = residual_norm * residual_norm;
        if (residual_norm == 0.0f || std::abs(x0) <= 1e-12f) {
            header.factor_dq = 0.0f;
            header.factor_vq = 0.0f;
            return;
        }

        header.factor_dq = -2.0f * residual_norm / x0 * inv_sqrt_code_dim_;
        header.factor_vq = header.factor_dq * static_cast<float>(code_sum);
    }

    uint32_t scalarCodeAccumulation(const QueryContext &query, const void *encoded) const {
        const uint8_t *code = codeBytes(encoded);
        uint32_t accum = 0;

        for (size_t i = 0; i < code_dim_; i += 2) {
            const uint8_t byte = code[i / 2];
            accum += static_cast<uint32_t>(query.byte_query[i]) *
                     static_cast<uint32_t>(byte & 0x0F);
            accum += static_cast<uint32_t>(query.byte_query[i + 1]) *
                     static_cast<uint32_t>(byte >> 4);
        }
        return accum;
    }

    float approximateDistanceFromAccumulation(
        const QueryContext &query,
        const EncodedHeader &header,
        uint32_t accumulation) const {
        const float result = static_cast<float>(accumulation);
        return header.norm_sqr + query.query_norm_sqr +
               query.query_norm * ((header.factor_dq * query.width * result) +
                                   (header.factor_vq * query.lower_val));
    }

    float queryDistancePrepared(const QueryContext &query, const void *encoded) const {
        return approximateDistanceFromAccumulation(
            query, loadHeader(encoded), scalarCodeAccumulation(query, encoded));
    }

    float distanceBetweenEncoded(const char *lhs, const char *rhs) const {
        std::vector<float> lhs_approx(code_dim_, 0.0f);
        std::vector<float> rhs_approx(code_dim_, 0.0f);
        decodeApproximate(lhs, lhs_approx);
        decodeApproximate(rhs, rhs_approx);

        float result = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            const float diff = lhs_approx[i] - rhs_approx[i];
            result += diff * diff;
        }
        return result;
    }

    void decodeApproximate(const void *encoded, std::vector<float> &decoded) const {
        const EncodedHeader header = loadHeader(encoded);
        const uint8_t *code = codeBytes(encoded);
        decoded.assign(code_dim_, 0.0f);

        // factor_dq = -2 * ||x-c0|| / x0 * (1 / sqrt(d)).
        // Therefore the reconstructed residual coefficient for each stored
        // 4-bit code value is ||x-c0|| / x0 * (1 / sqrt(d)) = -factor_dq / 2.
        const float scale = -0.5f * header.factor_dq;

        for (size_t i = 0; i < code_dim_; ++i) {
            const uint8_t byte = code[i / 2];
            const uint8_t qv = (i & 1U) == 0 ? (byte & 0x0F) : (byte >> 4);
            decoded[i] = rotated_global_center_[i] + scale * static_cast<float>(qv);
        }
    }

 public:
    explicit RaBitQSpace(size_t dim, size_t centroid_count = 1, uint32_t random_seed = 100)
        : dim_(dim),
          code_dim_(roundUp64(dim)),
          compact_code_bytes_(code_dim_ / 2),
          data_size_(sizeof(EncodedHeader) + compact_code_bytes_),
          inv_sqrt_code_dim_(1.0f / std::sqrt(static_cast<float>(code_dim_))),
          fstdistfunc_(exrabitqDistance),
          projection_(code_dim_ * code_dim_, 0.0f),
          global_center_(dim_, 0.0f),
          rotated_global_center_(code_dim_, 0.0f) {
        (void) centroid_count;
        setIdentityRotation();
        if (random_seed != 0) {
            setRandomRotation(random_seed);
        }
        refreshRotatedCenter();
    }

    void setIdentityRotation() {
        std::fill(projection_.begin(), projection_.end(), 0.0f);
        for (size_t i = 0; i < code_dim_; ++i) {
            projection_[i * code_dim_ + i] = 1.0f;
        }
        refreshRotatedCenter();
    }

    void setRandomRotation(uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        std::vector<float> basis(code_dim_ * code_dim_, 0.0f);
        std::vector<float> candidate(code_dim_, 0.0f);
        for (size_t col = 0; col < code_dim_; ++col) {
            float norm = 0.0f;
            for (size_t attempt = 0; attempt < 8; ++attempt) {
                std::fill(candidate.begin(), candidate.end(), 0.0f);
                for (size_t row = 0; row < code_dim_; ++row) {
                    candidate[row] = normal(rng);
                }

                for (size_t prev = 0; prev < col; ++prev) {
                    float dot = 0.0f;
                    for (size_t row = 0; row < code_dim_; ++row) {
                        dot += candidate[row] * basis[prev * code_dim_ + row];
                    }
                    for (size_t row = 0; row < code_dim_; ++row) {
                        candidate[row] -= dot * basis[prev * code_dim_ + row];
                    }
                }

                norm = 0.0f;
                for (size_t row = 0; row < code_dim_; ++row) {
                    norm += candidate[row] * candidate[row];
                }
                if (norm > 1e-12f) {
                    break;
                }
            }

            if (norm <= 1e-12f) {
                throw std::runtime_error("RaBitQSpace failed to generate an orthogonal projection");
            }

            const float inv_norm = 1.0f / std::sqrt(norm);
            for (size_t row = 0; row < code_dim_; ++row) {
                basis[col * code_dim_ + row] = candidate[row] * inv_norm;
            }
        }

        for (size_t row = 0; row < code_dim_; ++row) {
            for (size_t col = 0; col < code_dim_; ++col) {
                projection_[row * code_dim_ + col] = basis[col * code_dim_ + row];
            }
        }
        refreshRotatedCenter();
    }

    void setGlobalCenter(const float *raw_center) {
        std::copy(raw_center, raw_center + dim_, global_center_.begin());
        refreshRotatedCenter();
    }

    void setCentroids(const float *raw_centroids, size_t centroid_count) {
        if (centroid_count == 0) {
            throw std::invalid_argument("RaBitQSpace requires at least one center");
        }

        std::fill(global_center_.begin(), global_center_.end(), 0.0f);
        for (size_t centroid_id = 0; centroid_id < centroid_count; ++centroid_id) {
            const float *src = raw_centroids + centroid_id * dim_;
            for (size_t i = 0; i < dim_; ++i) {
                global_center_[i] += src[i];
            }
        }
        const float inv_count = 1.0f / static_cast<float>(centroid_count);
        for (float &value : global_center_) {
            value *= inv_count;
        }
        refreshRotatedCenter();
    }

    size_t get_dim() const {
        return dim_;
    }

    size_t get_code_dim() const {
        return code_dim_;
    }

    size_t get_compact_code_bytes() const {
        return compact_code_bytes_;
    }

    size_t get_centroid_count() const {
        return 1;
    }

    void saveState(std::ostream &output) const {
        const std::string magic = "EXRBTQ02";
        output.write(magic.data(), magic.size());

        const uint64_t dim = static_cast<uint64_t>(dim_);
        const uint64_t code_dim = static_cast<uint64_t>(code_dim_);
        output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        output.write(reinterpret_cast<const char *>(&code_dim), sizeof(code_dim));
        output.write(reinterpret_cast<const char *>(global_center_.data()), global_center_.size() * sizeof(float));
        output.write(reinterpret_cast<const char *>(projection_.data()), projection_.size() * sizeof(float));

        if (!output.good()) {
            throw std::runtime_error("RaBitQSpace failed to save ExRaBitQ state");
        }
    }

    void loadState(std::istream &input) {
        char magic[8];
        input.read(magic, sizeof(magic));
        if (!input.good() || std::string(magic, sizeof(magic)) != "EXRBTQ02") {
            throw std::runtime_error("RaBitQSpace invalid or old quantizer state; rebuild the index");
        }

        uint64_t stored_dim = 0;
        uint64_t stored_code_dim = 0;
        input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        input.read(reinterpret_cast<char *>(&stored_code_dim), sizeof(stored_code_dim));

        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read ExRaBitQ state header");
        }
        if (stored_dim != dim_ || stored_code_dim != code_dim_) {
            throw std::runtime_error("RaBitQSpace quantizer state is incompatible with this index");
        }

        global_center_.assign(dim_, 0.0f);
        projection_.assign(code_dim_ * code_dim_, 0.0f);
        input.read(reinterpret_cast<char *>(global_center_.data()), global_center_.size() * sizeof(float));
        input.read(reinterpret_cast<char *>(projection_.data()), projection_.size() * sizeof(float));

        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read ExRaBitQ state payload");
        }
        refreshRotatedCenter();
    }

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void *get_dist_func_param() override {
        return this;
    }

    const void *prepare_query(const void *query_data) override {
        const float *raw_query = static_cast<const float *>(query_data);
        QueryContext *query = new QueryContext();

        std::vector<float> residual(dim_, 0.0f);
        query->query_norm_sqr = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            residual[i] = raw_query[i] - global_center_[i];
            query->query_norm_sqr += residual[i] * residual[i];
        }

        query->query_norm = std::sqrt(query->query_norm_sqr);
        if (query->query_norm > 0.0f) {
            const float inv_norm = 1.0f / query->query_norm;
            for (float &value : residual) {
                value *= inv_norm;
            }
        }

        std::vector<float> rotated_query;
        rotate(residual.data(), rotated_query);

        float lower_val = std::numeric_limits<float>::max();
        float upper_val = std::numeric_limits<float>::lowest();
        for (float value : rotated_query) {
            lower_val = std::min(lower_val, value);
            upper_val = std::max(upper_val, value);
        }
        query->lower_val = lower_val;
        query->width = (upper_val - lower_val) / static_cast<float>((1 << kQueryBits) - 1);
        if (query->width <= 0.0f) {
            query->lower_val = 0.0f;
            query->width = 1.0f;
        }

        std::vector<uint8_t> byte_query(code_dim_, 0);
        const float inv_width = 1.0f / query->width;
        int32_t sumq = 0;
        for (size_t i = 0; i < code_dim_; ++i) {
            int value = static_cast<int>(std::lround((rotated_query[i] - query->lower_val) * inv_width));
            value = std::min((1 << kQueryBits) - 1, std::max(0, value));
            byte_query[i] = static_cast<uint8_t>(value);
            sumq += value;
        }
        query->sumq = sumq;

        query->byte_query.swap(byte_query);
        return query;
    }

    void release_query(const void *prepared_query) override {
        delete static_cast<const QueryContext *>(prepared_query);
    }

    float query_distance(const void *prepared_query, const void *data_point) override {
        return queryDistancePrepared(*static_cast<const QueryContext *>(prepared_query), data_point);
    }

    bool supports_batch_query_distance() const override {
        return true;
    }

    void batch_query_distance(
        const void *prepared_query,
        const void *const *data_points,
        size_t count,
        float *distances) override {
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        for (size_t i = 0; i < count; ++i) {
            distances[i] = queryDistancePrepared(query, data_points[i]);
        }
    }

    void encodeVector(const float *raw_vector, void *encoded_out) const {
        std::vector<float> residual(dim_, 0.0f);
        float residual_norm_sqr = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            residual[i] = raw_vector[i] - global_center_[i];
            residual_norm_sqr += residual[i] * residual[i];
        }

        const float residual_norm = std::sqrt(residual_norm_sqr);
        if (residual_norm > 0.0f) {
            const float inv_norm = 1.0f / residual_norm;
            for (float &value : residual) {
                value *= inv_norm;
            }
        }

        std::vector<float> rotated_unit;
        rotate(residual.data(), rotated_unit);

        EncodedHeader header{0.0f, 0.0f, 0.0f};
        uint8_t *code = codeBytes(encoded_out);
        exrabitqGlobalCode(rotated_unit.data(), residual_norm, code, header);
        std::memcpy(encoded_out, &header, sizeof(header));
    }

    std::vector<char> encodeVector(const float *raw_vector) const {
        std::vector<char> encoded(data_size_, 0);
        encodeVector(raw_vector, encoded.data());
        return encoded;
    }
};

}  // namespace hnswlib
