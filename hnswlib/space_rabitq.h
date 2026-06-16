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
    static constexpr size_t kBits = 8;
    static constexpr size_t kLevels = 1U << kBits;
    static constexpr uint32_t kMaxCode = static_cast<uint32_t>(kLevels - 1U);
    static constexpr float kFacRescale = static_cast<float>(1U << kBits);

    struct EncodedHeader {
        float norm_sqr;
        float long_scale;
    };

    struct ShortCodeFactors {
        float short_scale;
        float error_scale;
    };

    struct QueryContext {
        std::vector<float> rotated_residual;
        std::vector<float> short_lut;
        float query_norm = 0.0f;
        float query_norm_sqr = 0.0f;
        float half_sum_residual = 0.0f;
    };

 private:
    size_t dim_{0};
    size_t code_dim_{0};
    size_t compact_code_bytes_{0};
    size_t short_code_words_{0};
    size_t short_code_bytes_{0};
    size_t short_factor_bytes_{0};
    size_t data_size_{0};
    float inv_sqrt_code_dim_{1.0f};

    DISTFUNC<float> fstdistfunc_{nullptr};

    uint32_t random_seed_{100};
    std::vector<float> fht_signs_;
    std::vector<float> global_center_;

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

    const uint8_t *shortCodeBytes(const void *encoded) const {
        return reinterpret_cast<const uint8_t *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader) + compact_code_bytes_);
    }

    uint8_t *shortCodeBytes(void *encoded) const {
        return reinterpret_cast<uint8_t *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader) + compact_code_bytes_);
    }

    const ShortCodeFactors *shortFactors(const void *encoded) const {
        return reinterpret_cast<const ShortCodeFactors *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader) +
            compact_code_bytes_ + short_code_bytes_);
    }

    ShortCodeFactors *shortFactors(void *encoded) const {
        return reinterpret_cast<ShortCodeFactors *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader) +
            compact_code_bytes_ + short_code_bytes_);
    }

    void setShortCodeBit(uint8_t *short_code, size_t dim, bool value) const {
        const uint8_t mask = static_cast<uint8_t>(uint8_t{1} << (dim & 7U));
        uint8_t &byte = short_code[dim >> 3U];
        if (value) {
            byte |= mask;
        } else {
            byte &= static_cast<uint8_t>(~mask);
        }
    }

    void fastQuantizeAbs(const float *abs_unit_data, uint8_t *abs_code, float &ip_norm) const {
        constexpr double eps = 1e-5;
        constexpr int n_enum = 10;
        double max_o = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            max_o = std::max(max_o, static_cast<double>(abs_unit_data[i]));
        }
        if (max_o <= eps) {
            std::memset(abs_code, 0, compact_code_bytes_);
            ip_norm = 1.0f;
            return;
        }

        const double t_start = static_cast<double>((kMaxCode / 3U)) / max_o;
        const double t_end = (static_cast<double>(kMaxCode) + n_enum) / max_o;
        thread_local std::vector<int> cur_code;
        cur_code.assign(code_dim_, 0);

        double sqr_denominator = static_cast<double>(code_dim_) * 0.25;
        double numerator = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            cur_code[i] = static_cast<int>(t_start * abs_unit_data[i] + eps);
            cur_code[i] = std::min<int>(cur_code[i], static_cast<int>(kMaxCode));
            sqr_denominator += cur_code[i] * cur_code[i] + cur_code[i];
            numerator += (cur_code[i] + 0.5) * abs_unit_data[i];
        }

        std::priority_queue<
            std::pair<double, size_t>,
            std::vector<std::pair<double, size_t>>,
            std::greater<std::pair<double, size_t>>> next_t;
        for (size_t i = 0; i < code_dim_; ++i) {
            if (abs_unit_data[i] > eps) {
                next_t.emplace(static_cast<double>(cur_code[i] + 1) / abs_unit_data[i], i);
            }
        }

        double best_t = 0.0;
        double max_ip = 0.0;
        while (!next_t.empty()) {
            const double cur_t = next_t.top().first;
            const size_t update_id = next_t.top().second;
            next_t.pop();

            ++cur_code[update_id];
            const int update_code = cur_code[update_id];
            sqr_denominator += 2.0 * update_code;
            numerator += abs_unit_data[update_id];

            const double cur_ip = numerator / std::sqrt(sqr_denominator);
            if (cur_ip > max_ip) {
                max_ip = cur_ip;
                best_t = cur_t;
            }

            if (update_code < static_cast<int>(kMaxCode)) {
                const double candidate_t = static_cast<double>(update_code + 1) / abs_unit_data[update_id];
                if (candidate_t < t_end) {
                    next_t.emplace(candidate_t, update_id);
                }
            }
        }

        numerator = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            int value = static_cast<int>(best_t * abs_unit_data[i] + eps);
            value = std::min<int>(value, static_cast<int>(kMaxCode));
            abs_code[i] = static_cast<uint8_t>(value);
            numerator += (value + 0.5) * abs_unit_data[i];
        }

        ip_norm = numerator > eps ? static_cast<float>(1.0 / numerator) : 1.0f;
        if (!std::isfinite(ip_norm)) {
            ip_norm = 1.0f;
        }
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

    float dotUint8FloatAvxDispatch(const uint8_t *code, const float *query_rotated) const {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        return dotUint8FloatAvx512(code, query_rotated);
#elif defined(__AVX2__)
        return dotUint8FloatAvx2(code, query_rotated);
#else
        return dotUint8FloatScalar(code, query_rotated);
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

    float queryDistanceLong(const QueryContext &query, const void *encoded) const {
        const EncodedHeader header = loadHeader(encoded);
        const float short_ip = shortCodeIp(query, shortCodeBytes(encoded));
        return queryDistanceLongWithShortIp(query, encoded, header, short_ip);
    }

    float queryDistanceLongWithShortIp(
        const QueryContext &query,
        const void *encoded,
        const EncodedHeader &header,
        float short_ip) const {
        if (header.long_scale <= 0.0f || !std::isfinite(header.long_scale)) {
            return header.norm_sqr + query.query_norm_sqr;
        }
        const float long_ip = dotUint8FloatAvxDispatch(codeBytes(encoded), query.rotated_residual.data());
        const float signed_long_ip =
            kFacRescale * short_ip + long_ip - (kFacRescale - 1.0f) * query.half_sum_residual;
        const float estimated_inner = header.long_scale * signed_long_ip;
        return header.norm_sqr + query.query_norm_sqr -
               estimated_inner;
    }

    float queryDistanceLowerBound(const QueryContext &query, const void *encoded) const {
        const EncodedHeader header = loadHeader(encoded);
        const ShortCodeFactors factors = *shortFactors(encoded);
        const float ip_xb_q = shortCodeIp(query, shortCodeBytes(encoded));
        return queryDistanceLowerBoundWithShortIp(query, header, factors, ip_xb_q);
    }

    float queryDistanceLowerBoundWithShortIp(
        const QueryContext &query,
        const EncodedHeader &header,
        const ShortCodeFactors &factors,
        float ip_xb_q) const {
        const float compensated_ip = ip_xb_q * factors.short_scale;
        const float err = factors.error_scale * query.query_norm;
        return header.norm_sqr + query.query_norm_sqr - (compensated_ip + err);
    }

    float shortCodeIp(const QueryContext &query, const uint8_t *data_short_code) const {
        float ip_xb_q = 0.0f;
        for (size_t i = 0; i < short_code_bytes_; ++i) {
            ip_xb_q += query.short_lut[i * 256U + data_short_code[i]];
        }
        return ip_xb_q - query.half_sum_residual;
    }

    float distanceBetweenEncoded(const char *lhs, const char *rhs) const {
        const EncodedHeader lhs_header = loadHeader(lhs);
        const EncodedHeader rhs_header = loadHeader(rhs);
        const uint8_t *lhs_code = codeBytes(lhs);
        const uint8_t *rhs_code = codeBytes(rhs);
        const uint8_t *lhs_short = shortCodeBytes(lhs);
        const uint8_t *rhs_short = shortCodeBytes(rhs);

        double code_ip = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            const bool lhs_positive = (lhs_short[i >> 3U] & (uint8_t{1} << (i & 7U))) != 0;
            const bool rhs_positive = (rhs_short[i >> 3U] & (uint8_t{1} << (i & 7U))) != 0;
            const float lhs_mag = lhs_positive
                                      ? static_cast<float>(lhs_code[i]) + 0.5f
                                      : static_cast<float>(kMaxCode - lhs_code[i]) + 0.5f;
            const float rhs_mag = rhs_positive
                                      ? static_cast<float>(rhs_code[i]) + 0.5f
                                      : static_cast<float>(kMaxCode - rhs_code[i]) + 0.5f;
            code_ip += (lhs_positive == rhs_positive ? 1.0 : -1.0) *
                       static_cast<double>(lhs_mag) * static_cast<double>(rhs_mag);
        }

        const double residual_ip =
            0.25 * static_cast<double>(lhs_header.long_scale) *
            static_cast<double>(rhs_header.long_scale) * code_ip;
        return static_cast<float>(
            static_cast<double>(lhs_header.norm_sqr) + static_cast<double>(rhs_header.norm_sqr) -
            2.0 * residual_ip);
    }

 public:
    explicit RaBitQSpace(size_t dim, size_t centroid_count = 1, uint32_t random_seed = 100)
        : dim_(dim),
          code_dim_(roundUp64(dim)),
          compact_code_bytes_(code_dim_),
          short_code_words_(code_dim_ / 64U),
          short_code_bytes_(short_code_words_ * sizeof(uint64_t)),
          short_factor_bytes_(sizeof(ShortCodeFactors)),
          data_size_(sizeof(EncodedHeader) + compact_code_bytes_ +
                     short_code_bytes_ + short_factor_bytes_),
          inv_sqrt_code_dim_(1.0f / std::sqrt(static_cast<float>(code_dim_))),
          fstdistfunc_(exrabitqDistance),
          random_seed_(random_seed),
          fht_signs_(code_dim_, inv_sqrt_code_dim_),
          global_center_(dim_, 0.0f) {
        (void) centroid_count;
        if (random_seed != 0) {
            setRandomRotation(random_seed);
        } else {
            setIdentityRotation();
        }
    }

    void setIdentityRotation() {
        random_seed_ = 0;
        std::fill(fht_signs_.begin(), fht_signs_.end(), inv_sqrt_code_dim_);
    }

    void setRandomRotation(uint32_t seed) {
        random_seed_ = seed;
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> bernoulli(0, 1);
        for (size_t i = 0; i < code_dim_; ++i) {
            fht_signs_[i] = (bernoulli(rng) ? 1.0f : -1.0f) * inv_sqrt_code_dim_;
        }
    }

    void setGlobalCenter(const float *raw_center) {
        std::copy(raw_center, raw_center + dim_, global_center_.begin());
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
        const std::string magic = "EXRBTQ15";
        output.write(magic.data(), magic.size());

        const uint64_t dim = static_cast<uint64_t>(dim_);
        const uint64_t code_dim = static_cast<uint64_t>(code_dim_);
        const uint32_t bits = static_cast<uint32_t>(kBits);
        const uint64_t encoded_header_size = static_cast<uint64_t>(sizeof(EncodedHeader));
        const uint64_t short_factor_size = static_cast<uint64_t>(sizeof(ShortCodeFactors));
        const uint64_t long_code_bytes = static_cast<uint64_t>(compact_code_bytes_);
        const uint64_t short_code_bytes = static_cast<uint64_t>(short_code_bytes_);
        output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        output.write(reinterpret_cast<const char *>(&code_dim), sizeof(code_dim));
        output.write(reinterpret_cast<const char *>(&bits), sizeof(bits));
        output.write(reinterpret_cast<const char *>(&encoded_header_size), sizeof(encoded_header_size));
        output.write(reinterpret_cast<const char *>(&short_factor_size), sizeof(short_factor_size));
        output.write(reinterpret_cast<const char *>(&long_code_bytes), sizeof(long_code_bytes));
        output.write(reinterpret_cast<const char *>(&short_code_bytes), sizeof(short_code_bytes));
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
        if (!input.good() || std::string(magic, sizeof(magic)) != "EXRBTQ15") {
            throw std::runtime_error(
                "Old or incompatible RaBitQ index format. Please rebuild the index.");
        }

        uint64_t stored_dim = 0;
        uint64_t stored_code_dim = 0;
        uint32_t stored_bits = 0;
        uint64_t stored_header_size = 0;
        uint64_t stored_factor_size = 0;
        uint64_t stored_long_code_bytes = 0;
        uint64_t stored_short_code_bytes = 0;
        input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        input.read(reinterpret_cast<char *>(&stored_code_dim), sizeof(stored_code_dim));
        input.read(reinterpret_cast<char *>(&stored_bits), sizeof(stored_bits));
        input.read(reinterpret_cast<char *>(&stored_header_size), sizeof(stored_header_size));
        input.read(reinterpret_cast<char *>(&stored_factor_size), sizeof(stored_factor_size));
        input.read(reinterpret_cast<char *>(&stored_long_code_bytes), sizeof(stored_long_code_bytes));
        input.read(reinterpret_cast<char *>(&stored_short_code_bytes), sizeof(stored_short_code_bytes));

        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read ExRaBitQ state header");
        }
        if (stored_dim != dim_ ||
            stored_code_dim != code_dim_ ||
            stored_bits != kBits ||
            stored_header_size != sizeof(EncodedHeader) ||
            stored_factor_size != sizeof(ShortCodeFactors) ||
            stored_long_code_bytes != compact_code_bytes_ ||
            stored_short_code_bytes != short_code_bytes_) {
            throw std::runtime_error("Old or incompatible RaBitQ index format. Please rebuild the index.");
        }

        global_center_.assign(dim_, 0.0f);
        fht_signs_.assign(code_dim_, 0.0f);
        input.read(reinterpret_cast<char *>(&random_seed_), sizeof(random_seed_));
        input.read(reinterpret_cast<char *>(global_center_.data()), global_center_.size() * sizeof(float));
        input.read(reinterpret_cast<char *>(fht_signs_.data()), fht_signs_.size() * sizeof(float));

        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read ExRaBitQ state payload");
        }
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
        thread_local std::vector<float> residual;
        residual.assign(dim_, 0.0f);
        query->short_lut.assign(short_code_bytes_ * 256U, 0.0f);

        query->query_norm_sqr = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            residual[i] = raw_query[i] - global_center_[i];
            query->query_norm_sqr += residual[i] * residual[i];
        }

        query->rotated_residual.assign(code_dim_, 0.0f);
        rotate(residual.data(), query->rotated_residual);
        query->half_sum_residual = 0.0f;
        for (float value : query->rotated_residual) {
            query->half_sum_residual += value;
        }
        query->half_sum_residual *= 0.5f;
        for (size_t byte_id = 0; byte_id < short_code_bytes_; ++byte_id) {
            const size_t base_dim = byte_id * 8U;
            for (size_t mask = 1; mask < 256U; ++mask) {
                const size_t bit = static_cast<size_t>(__builtin_ctz(static_cast<unsigned>(mask)));
                const uint8_t previous = static_cast<uint8_t>(mask & (mask - 1U));
                const float addend = base_dim + bit < code_dim_ ? query->rotated_residual[base_dim + bit] : 0.0f;
                query->short_lut[byte_id * 256U + mask] =
                    query->short_lut[byte_id * 256U + previous] + addend;
            }
        }

        query->query_norm = std::sqrt(query->query_norm_sqr);

        return query;
    }

    void release_query(const void *prepared_query) override {
        (void) prepared_query;
    }

    float query_distance(const void *prepared_query, const void *data_point) override {
        return queryDistanceLong(*static_cast<const QueryContext *>(prepared_query), data_point);
    }

    bool supports_query_distance_lower_bound() const override {
        return true;
    }

    float query_distance_lower_bound(const void *prepared_query, const void *data_point) override {
        return queryDistanceLowerBound(*static_cast<const QueryContext *>(prepared_query), data_point);
    }

    bool query_distance_if_lower_bound_below(
        const void *prepared_query,
        const void *data_point,
        float threshold,
        float *distance) override {
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        const EncodedHeader header = loadHeader(data_point);
        const ShortCodeFactors factors = *shortFactors(data_point);
        const float short_ip = shortCodeIp(query, shortCodeBytes(data_point));
        const float lower_bound = queryDistanceLowerBoundWithShortIp(query, header, factors, short_ip);
        if (lower_bound >= threshold) {
            return false;
        }
        *distance = queryDistanceLongWithShortIp(query, data_point, header, short_ip);
        return true;
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
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        for (size_t i = 0; i < count; ++i) {
            distances[i] = queryDistanceLong(query, data_points[i]);
        }
    }

    void batch_query_distance_lower_bound(
        const void *prepared_query,
        const void *const *data_points,
        size_t count,
        float *lower_bounds) override {
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        for (size_t i = 0; i < count; ++i) {
            lower_bounds[i] = queryDistanceLowerBound(query, data_points[i]);
        }
    }

    size_t batch_query_distance_if_lower_bound_below(
        const void *prepared_query,
        const void *const *data_points,
        size_t count,
        float threshold,
        bool use_threshold,
        float *distances,
        size_t *survivor_indices) override {
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        size_t survivor_count = 0;
        for (size_t i = 0; i < count; ++i) {
            const void *encoded = data_points[i];
            const EncodedHeader header = loadHeader(encoded);
            const float short_ip = shortCodeIp(query, shortCodeBytes(encoded));
            if (use_threshold) {
                const ShortCodeFactors factors = *shortFactors(encoded);
                const float lower_bound =
                    queryDistanceLowerBoundWithShortIp(query, header, factors, short_ip);
                if (lower_bound > threshold) {
                    continue;
                }
            }
            distances[survivor_count] = queryDistanceLongWithShortIp(query, encoded, header, short_ip);
            survivor_indices[survivor_count] = i;
            ++survivor_count;
        }
        return survivor_count;
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

        EncodedHeader header{0.0f, 0.0f};
        uint8_t *code = codeBytes(encoded_out);
        uint8_t *short_code = shortCodeBytes(encoded_out);
        ShortCodeFactors *factors = shortFactors(encoded_out);
        header.norm_sqr = residual_norm * residual_norm;
        std::memset(short_code, 0, short_code_bytes_);
        std::memset(code, 0, compact_code_bytes_);
        *factors = ShortCodeFactors{0.0f, 0.0f};

        double o_obar = 0.0;
        double half_l1_norm = 0.0;
        const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(code_dim_));
        for (size_t i = 0; i < code_dim_; ++i) {
            const bool bit = rotated_unit[i] > 0.0f;
            setShortCodeBit(short_code, i, bit);
            const double sign = bit ? 1.0 : -1.0;
            o_obar += static_cast<double>(rotated_unit[i]) * sign * inv_sqrt_d;
            half_l1_norm += 0.5 * std::abs(static_cast<double>(rotated_unit[i]));
        }
        if (residual_norm > 0.0f && half_l1_norm > 1e-12) {
            factors->short_scale = static_cast<float>(2.0 * residual_norm / half_l1_norm);
            if (!std::isfinite(o_obar)) {
                o_obar = 0.8;
            }
            o_obar = std::min(0.999999, std::max(1e-6, o_obar));
            const double o2 = o_obar * o_obar;
            const double fac_err = 2.0 / std::sqrt(static_cast<double>(code_dim_ - 1U));
            factors->error_scale = static_cast<float>(
                std::sqrt(std::max(0.0, (1.0 - o2) / o2)) * fac_err * 2.0 * residual_norm);
        }

        thread_local std::vector<float> abs_unit;
        thread_local std::vector<uint8_t> abs_code;
        abs_unit.assign(code_dim_, 0.0f);
        abs_code.assign(code_dim_, 0);
        for (size_t i = 0; i < code_dim_; ++i) {
            abs_unit[i] = std::abs(rotated_unit[i]);
        }
        float ip_norm = 1.0f;
        fastQuantizeAbs(abs_unit.data(), abs_code.data(), ip_norm);
        header.long_scale = 2.0f * residual_norm * ip_norm;
        for (size_t i = 0; i < code_dim_; ++i) {
            const bool positive = rotated_unit[i] > 0.0f;
            code[i] = positive ? abs_code[i] : static_cast<uint8_t>(kMaxCode - abs_code[i]);
        }
        std::memcpy(encoded_out, &header, sizeof(header));
    }

    std::vector<char> encodeVector(const float *raw_vector) const {
        std::vector<char> encoded(data_size_, 0);
        encodeVector(raw_vector, encoded.data());
        return encoded;
    }
};

}  // namespace hnswlib
