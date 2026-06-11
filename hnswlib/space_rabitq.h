#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hnswlib.h"

namespace hnswlib {

class RaBitQSpace : public SpaceInterface<float> {
 public:
    static constexpr size_t kBits = 8;
    static constexpr size_t kLevels = 1U << kBits;
    static constexpr uint32_t kMaxCode = static_cast<uint32_t>(kLevels - 1U);
    static constexpr float kOffset = 0.5f * static_cast<float>(kMaxCode);
    static constexpr size_t kBaseBits = kBits;
    static constexpr uint8_t kBaseMax = static_cast<uint8_t>(kMaxCode);
    static constexpr float kCodeOffset = kOffset;
    static constexpr size_t kFactorCount = 2;

    struct EncodedHeader {
        float norm_sqr;
        float norm;
        float inv_dot_y_o_prime;
    };

    struct QueryContext {
        std::vector<float> residual;
        std::vector<float> rotated;
        float query_norm = 0.0f;
        float query_norm_sqr = 0.0f;
        float sum_q = 0.0f;
    };

 private:
    size_t dim_{0};
    size_t code_dim_{0};
    size_t compact_code_bytes_{0};
    size_t data_size_{0};
    float inv_sqrt_code_dim_{1.0f};

    DISTFUNC<float> fstdistfunc_{nullptr};

    uint32_t random_seed_{100};
    std::vector<float> fht_signs_;
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

    uint32_t getCode(const uint8_t *code, size_t dim) const {
        return static_cast<uint32_t>(code[dim]);
    }

    void setCode(uint8_t *code, size_t dim, uint32_t value) const {
        code[dim] = static_cast<uint8_t>(std::min<uint32_t>(kMaxCode, value));
    }

    uint32_t quantizeCode(double value) const {
        const double rounded = std::round(value + static_cast<double>(kOffset));
        if (rounded <= 0.0) {
            return 0;
        }
        if (rounded >= static_cast<double>(kMaxCode)) {
            return kMaxCode;
        }
        return static_cast<uint32_t>(rounded);
    }

    float centeredCodeValue(uint32_t value) const {
        return static_cast<float>(value) - kOffset;
    }

    void hadamard(std::vector<float> &values) const {
        for (size_t step = 1; step < code_dim_; step <<= 1U) {
            for (size_t block = 0; block < code_dim_; block += (step << 1U)) {
                for (size_t i = 0; i < step; ++i) {
                    const float a = values[block + i];
                    const float b = values[block + step + i];
                    values[block + i] = a + b;
                    values[block + step + i] = a - b;
                }
            }
        }
    }

    void rotate(const float *raw_vector, std::vector<float> &rotated) const {
        rotated.assign(code_dim_, 0.0f);
        if (random_seed_ == 0) {
            std::copy(raw_vector, raw_vector + dim_, rotated.begin());
            return;
        }
        for (size_t i = 0; i < dim_; ++i) {
            rotated[i] = raw_vector[i] * fht_signs_[i];
        }
        hadamard(rotated);
    }

    void refreshRotatedCenter() {
        rotate(global_center_.data(), rotated_global_center_);
    }

    void applyCodeUpdate(
        size_t dim,
        uint32_t new_code,
        const float *rotated_unit_data,
        std::vector<uint32_t> &current_code,
        double &dot_y_o_prime,
        double &norm_y_sqr) const {
        new_code = std::min<uint32_t>(kMaxCode, new_code);
        const uint32_t old_code = current_code[dim];
        if (new_code == old_code) {
            return;
        }

        const double o = static_cast<double>(rotated_unit_data[dim]);
        const double old_y = static_cast<double>(old_code) - static_cast<double>(kOffset);
        const double new_y = static_cast<double>(new_code) - static_cast<double>(kOffset);
        dot_y_o_prime += (new_y - old_y) * o;
        norm_y_sqr += new_y * new_y - old_y * old_y;
        current_code[dim] = new_code;
    }

    void exrabitqGlobalCode(
        const float *rotated_unit_data,
        float residual_norm,
        uint8_t *code,
        EncodedHeader &header) const {
        constexpr double eps = 1e-12;
        constexpr double t_group_rel_eps = 1e-12;

        std::memset(code, 0, compact_code_bytes_);
        header.norm_sqr = residual_norm * residual_norm;
        header.norm = residual_norm;
        header.inv_dot_y_o_prime = 0.0f;

        if (residual_norm == 0.0f) {
            return;
        }

        thread_local std::vector<uint32_t> current_code;
        thread_local std::vector<uint32_t> best_code;
        thread_local std::vector<std::pair<double, size_t>> breakpoints;

        current_code.assign(code_dim_, quantizeCode(0.0));
        best_code = current_code;
        breakpoints.clear();
        breakpoints.reserve(code_dim_ * static_cast<size_t>(kMaxCode));

        double dot_y_o_prime = 0.0;
        double norm_y_sqr = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            const double o = static_cast<double>(rotated_unit_data[i]);
            const double y = static_cast<double>(current_code[i]) - static_cast<double>(kOffset);
            dot_y_o_prime += y * o;
            norm_y_sqr += y * y;

            if (std::abs(o) <= eps) {
                continue;
            }
            for (int k = 0; k < static_cast<int>(kMaxCode); ++k) {
                const double boundary = static_cast<double>(k) + 0.5;
                const double t = (boundary - static_cast<double>(kOffset)) / o;
                if (std::isfinite(t) && t > 0.0) {
                    breakpoints.emplace_back(t, i);
                }
            }
        }

        double best_score = norm_y_sqr > eps ? dot_y_o_prime / std::sqrt(norm_y_sqr) : 0.0;

        for (size_t i = 0; i < code_dim_; ++i) {
            if (rotated_unit_data[i] < 0.0f && current_code[i] > 0) {
                applyCodeUpdate(
                    i,
                    current_code[i] - 1U,
                    rotated_unit_data,
                    current_code,
                    dot_y_o_prime,
                    norm_y_sqr);
            }
        }
        if (norm_y_sqr > eps) {
            const double score = dot_y_o_prime / std::sqrt(norm_y_sqr);
            if (score > best_score) {
                best_score = score;
                best_code = current_code;
            }
        }

        std::sort(breakpoints.begin(), breakpoints.end());
        size_t pos = 0;
        while (pos < breakpoints.size()) {
            const double cur_t = breakpoints[pos].first;
            const double group_eps = std::max(1e-15, std::abs(cur_t) * t_group_rel_eps);
            while (pos < breakpoints.size() && breakpoints[pos].first <= cur_t + group_eps) {
                const size_t dim = breakpoints[pos].second;
                const double o = static_cast<double>(rotated_unit_data[dim]);
                const uint32_t new_code = quantizeCode(breakpoints[pos].first * o);
                applyCodeUpdate(dim, new_code, rotated_unit_data, current_code, dot_y_o_prime, norm_y_sqr);
                ++pos;
            }

            if (norm_y_sqr > eps) {
                const double score = dot_y_o_prime / std::sqrt(norm_y_sqr);
                if (score > best_score) {
                    best_score = score;
                    best_code = current_code;
                }
            }
        }

        for (size_t i = 0; i < code_dim_; ++i) {
            setCode(code, i, best_code[i]);
        }

        double final_dot_y_o_prime = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            const double y = static_cast<double>(getCode(code, i)) - static_cast<double>(kOffset);
            final_dot_y_o_prime += y * static_cast<double>(rotated_unit_data[i]);
        }

        if (final_dot_y_o_prime <= eps || !std::isfinite(final_dot_y_o_prime)) {
            return;
        }
        header.inv_dot_y_o_prime = static_cast<float>(1.0 / final_dot_y_o_prime);
    }

    float dotUint8Float(const QueryContext &query, const void *encoded) const {
        const uint8_t *code = codeBytes(encoded);
#if defined(__AVX512F__) && defined(__AVX512BW__)
        return dotUint8FloatAvx512(code, query.rotated.data());
#elif defined(__AVX2__)
        return dotUint8FloatAvx2(code, query.rotated.data());
#else
        return dotUint8FloatScalar(code, query.rotated.data());
#endif
    }

    float dotUint8FloatScalar(const uint8_t *code, const float *query_rotated) const {
        float result = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            result += static_cast<float>(code[i]) * query_rotated[i];
        }
        return result;
    }

#if defined(__AVX2__)
    float dotUint8FloatAvx2(const uint8_t *code, const float *query_rotated) const {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= code_dim_; i += 8) {
            const __m128i code8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(code + i));
            const __m256 code_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code8));
            const __m256 query_f = _mm256_loadu_ps(query_rotated + i);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(code_f, query_f));
        }

        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sum);
        float result = 0.0f;
        for (float lane : lanes) {
            result += lane;
        }
        for (; i < code_dim_; ++i) {
            result += static_cast<float>(code[i]) * query_rotated[i];
        }
        return result;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    float dotUint8FloatAvx512(const uint8_t *code, const float *query_rotated) const {
        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= code_dim_; i += 16) {
            const __m128i code8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(code + i));
            const __m512 code_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(code8));
            const __m512 query_f = _mm512_loadu_ps(query_rotated + i);
            sum = _mm512_fmadd_ps(code_f, query_f, sum);
        }

        float result = _mm512_reduce_add_ps(sum);
        for (; i < code_dim_; ++i) {
            result += static_cast<float>(code[i]) * query_rotated[i];
        }
        return result;
    }
#endif

    float queryDistancePrepared(const QueryContext &query, const void *encoded) const {
        const EncodedHeader header = loadHeader(encoded);
        if (header.inv_dot_y_o_prime <= 0.0f || !std::isfinite(header.inv_dot_y_o_prime)) {
            return header.norm_sqr + query.query_norm_sqr;
        }
        const float dot_centered = dotUint8Float(query, encoded) - kOffset * query.sum_q;
        const float estimated_inner = dot_centered * header.inv_dot_y_o_prime;
        return header.norm_sqr + query.query_norm_sqr -
               2.0f * header.norm * query.query_norm * estimated_inner;
    }

    float distanceBetweenEncoded(const char *lhs, const char *rhs) const {
        thread_local std::vector<float> lhs_approx;
        thread_local std::vector<float> rhs_approx;
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

        float code_norm_sqr = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            const float centered = centeredCodeValue(getCode(code, i));
            code_norm_sqr += centered * centered;
        }
        const float scale = code_norm_sqr > 0.0f
                                ? std::sqrt(header.norm_sqr / code_norm_sqr)
                                : 0.0f;

        for (size_t i = 0; i < code_dim_; ++i) {
            const float centered = centeredCodeValue(getCode(code, i));
            decoded[i] = rotated_global_center_[i] + scale * centered;
        }
    }

 public:
    explicit RaBitQSpace(size_t dim, size_t centroid_count = 1, uint32_t random_seed = 100)
        : dim_(dim),
          code_dim_(roundUp64(dim)),
          compact_code_bytes_(code_dim_),
          data_size_(sizeof(EncodedHeader) + compact_code_bytes_),
          inv_sqrt_code_dim_(1.0f / std::sqrt(static_cast<float>(code_dim_))),
          fstdistfunc_(exrabitqDistance),
          random_seed_(random_seed),
          fht_signs_(code_dim_, inv_sqrt_code_dim_),
          global_center_(dim_, 0.0f),
          rotated_global_center_(code_dim_, 0.0f) {
        (void) centroid_count;
        if (random_seed != 0) {
            setRandomRotation(random_seed);
        } else {
            setIdentityRotation();
        }
        refreshRotatedCenter();
    }

    void setIdentityRotation() {
        random_seed_ = 0;
        std::fill(fht_signs_.begin(), fht_signs_.end(), inv_sqrt_code_dim_);
        refreshRotatedCenter();
    }

    void setRandomRotation(uint32_t seed) {
        random_seed_ = seed;
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> bernoulli(0, 1);
        for (size_t i = 0; i < code_dim_; ++i) {
            fht_signs_[i] = (bernoulli(rng) ? 1.0f : -1.0f) * inv_sqrt_code_dim_;
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
        const std::string magic = "EXRBTQ11";
        output.write(magic.data(), magic.size());

        const uint64_t dim = static_cast<uint64_t>(dim_);
        const uint64_t code_dim = static_cast<uint64_t>(code_dim_);
        output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        output.write(reinterpret_cast<const char *>(&code_dim), sizeof(code_dim));
        output.write(reinterpret_cast<const char *>(&random_seed_), sizeof(random_seed_));
        output.write(reinterpret_cast<const char *>(global_center_.data()), global_center_.size() * sizeof(float));
        output.write(reinterpret_cast<const char *>(fht_signs_.data()), fht_signs_.size() * sizeof(float));

        if (!output.good()) {
            throw std::runtime_error("RaBitQSpace failed to save ExRaBitQ state");
        }
    }

    void loadState(std::istream &input) {
        char magic[8];
        input.read(magic, sizeof(magic));
        if (!input.good() || std::string(magic, sizeof(magic)) != "EXRBTQ11") {
            throw std::runtime_error(
                "Old RaBitQ index format is incompatible with optimized B=8 layout. Please rebuild.");
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
        fht_signs_.assign(code_dim_, 0.0f);
        input.read(reinterpret_cast<char *>(&random_seed_), sizeof(random_seed_));
        input.read(reinterpret_cast<char *>(global_center_.data()), global_center_.size() * sizeof(float));
        input.read(reinterpret_cast<char *>(fht_signs_.data()), fht_signs_.size() * sizeof(float));

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
        thread_local QueryContext query_storage;
        QueryContext *query = &query_storage;
        query->residual.assign(dim_, 0.0f);
        query->rotated.assign(code_dim_, 0.0f);

        query->query_norm_sqr = 0.0f;
        query->sum_q = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            query->residual[i] = raw_query[i] - global_center_[i];
            query->query_norm_sqr += query->residual[i] * query->residual[i];
        }

        query->query_norm = std::sqrt(query->query_norm_sqr);
        if (query->query_norm > 0.0f) {
            const float inv_norm = 1.0f / query->query_norm;
            for (float &value : query->residual) {
                value *= inv_norm;
            }
        }

        rotate(query->residual.data(), query->rotated);
        for (float value : query->rotated) {
            query->sum_q += value;
        }

        return query;
    }

    void release_query(const void *prepared_query) override {
        (void) prepared_query;
    }

    float query_distance(const void *prepared_query, const void *data_point) override {
        return queryDistancePrepared(*static_cast<const QueryContext *>(prepared_query), data_point);
    }

    float result_distance(const void *prepared_query, const void *data_point) override {
        return query_distance(prepared_query, data_point);
    }

    bool supports_batch_query_distance() const override {
        return true;
    }

    void batch_query_distance(
        const void *prepared_query,
        const void *const *data_points,
        size_t count,
        float *distances) override {
        for (size_t i = 0; i < count; ++i) {
            distances[i] = query_distance(prepared_query, data_points[i]);
        }
    }

    void encodeVector(const float *raw_vector, void *encoded_out) const {
        thread_local std::vector<float> residual;
        residual.assign(dim_, 0.0f);
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

        thread_local std::vector<float> rotated_unit;
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
