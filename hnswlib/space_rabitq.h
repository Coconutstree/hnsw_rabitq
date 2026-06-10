#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <istream>
#include <mutex>
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
    static constexpr float kExtendedBias = static_cast<float>(1U << kBaseBits) - 0.5f;
    static constexpr int kQueryBits = 6;
    static constexpr size_t kFactorCount = 3;

    struct EncodedHeader {
        float norm_sqr;
        float factor_dq;  // ExRaBitQ xipnorm: 2 * ||x-c|| / <|o|, o_bar + 0.5>.
        float factor_vq;
    };

    struct QueryContext {
        std::vector<uint8_t> byte_query;
        std::vector<uint8_t> raw_query_u8;
        std::vector<float> residual;
        std::vector<float> rotated;
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
    size_t sign_code_bytes_{0};
    size_t data_size_{0};
    float inv_sqrt_code_dim_{1.0f};

    DISTFUNC<float> fstdistfunc_{nullptr};

    uint32_t random_seed_{100};
    std::vector<float> fht_signs_;
    std::vector<float> global_center_;
    std::vector<float> rotated_global_center_;
    std::vector<uint8_t> raw_store_;
    mutable std::mutex raw_store_mutex_;

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

    const uint8_t *signBytes(const void *encoded) const {
        return codeBytes(encoded) + compact_code_bytes_;
    }

    uint8_t *signBytes(void *encoded) const {
        return codeBytes(encoded) + compact_code_bytes_;
    }

    static bool signBit(const uint8_t *sign_code, size_t dim) {
        return ((sign_code[dim >> 3U] >> (dim & 7U)) & 1U) != 0;
    }

    static void setSignBit(uint8_t *sign_code, size_t dim) {
        sign_code[dim >> 3U] |= static_cast<uint8_t>(1U << (dim & 7U));
    }

    static std::vector<uint8_t> &pendingRawStorage() {
        thread_local std::vector<uint8_t> pending_raw;
        return pending_raw;
    }

    uint8_t compactCodeValue(const uint8_t *code, size_t dim) const {
        const uint8_t byte = code[dim >> 1U];
        return (dim & 1U) == 0 ? static_cast<uint8_t>(byte & 0x0F)
                               : static_cast<uint8_t>(byte >> 4);
    }

    void encodeRawVector(const float *raw_vector, uint8_t *raw_out) const {
        for (size_t i = 0; i < dim_; ++i) {
            int raw_value = static_cast<int>(std::lround(raw_vector[i]));
            raw_value = std::min(255, std::max(0, raw_value));
            raw_out[i] = static_cast<uint8_t>(raw_value);
        }
    }

    const uint8_t *rawByInternalId(size_t internal_id) const {
        const size_t offset = internal_id * dim_;
        if (dim_ == 0 || offset + dim_ > raw_store_.size()) {
            return nullptr;
        }
        return raw_store_.data() + offset;
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

    void exrabitqGlobalCode(
        const float *rotated_unit_data,
        float residual_norm,
        uint8_t *compact_code,
        uint8_t *sign_code,
        EncodedHeader &header) const {
        constexpr double eps = 1e-5;
        constexpr int n_enum = 10;

        std::memset(compact_code, 0, compact_code_bytes_);
        std::memset(sign_code, 0, sign_code_bytes_);
        header.norm_sqr = residual_norm * residual_norm;
        header.factor_dq = 0.0f;
        header.factor_vq = 0.0f;

        double max_o = -1.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            max_o = std::max(max_o, static_cast<double>(std::abs(rotated_unit_data[i])));
        }
        if (residual_norm == 0.0f || max_o <= eps) {
            return;
        }

        double t_start = static_cast<double>(kBaseMax / 3U) / max_o;
        double t_end = static_cast<double>(kBaseMax + n_enum) / max_o;

        thread_local std::vector<int> cur_o_bar;
        cur_o_bar.assign(code_dim_, 0);
        double sqr_denominator = static_cast<double>(code_dim_) * 0.25;
        double numerator = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            const double abs_o = static_cast<double>(std::abs(rotated_unit_data[i]));
            cur_o_bar[i] = static_cast<int>(t_start * abs_o + eps);
            cur_o_bar[i] = std::min<int>(kBaseMax, std::max(0, cur_o_bar[i]));
            sqr_denominator += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
            numerator += (cur_o_bar[i] + 0.5) * abs_o;
        }

        thread_local std::vector<std::pair<double, size_t>> next_t;
        next_t.clear();
        next_t.reserve(code_dim_);

        for (size_t i = 0; i < code_dim_; ++i) {
            const double abs_o = static_cast<double>(std::abs(rotated_unit_data[i]));
            if (abs_o > eps) {
                next_t.emplace_back((cur_o_bar[i] + 1) / abs_o, i);
            }
        }
        std::make_heap(next_t.begin(), next_t.end(), std::greater<std::pair<double, size_t>>());

        double max_ip = 0.0;
        double best_t = t_start;
        while (!next_t.empty()) {
            std::pop_heap(next_t.begin(), next_t.end(), std::greater<std::pair<double, size_t>>());
            const auto next_pair = next_t.back();
            next_t.pop_back();
            const double cur_t = next_pair.first;
            const size_t update_id = next_pair.second;

            if (cur_t > t_end) {
                break;
            }

            cur_o_bar[update_id]++;
            const int update_o_bar = cur_o_bar[update_id];
            sqr_denominator += 2 * update_o_bar;
            numerator += std::abs(rotated_unit_data[update_id]);

            const double cur_ip = numerator / std::sqrt(sqr_denominator);
            if (cur_ip > max_ip) {
                max_ip = cur_ip;
                best_t = cur_t;
            }

            if (update_o_bar < static_cast<int>(kBaseMax)) {
                const double t_next = (update_o_bar + 1) /
                                      static_cast<double>(std::abs(rotated_unit_data[update_id]));
                if (t_next < t_end) {
                    next_t.emplace_back(t_next, update_id);
                    std::push_heap(next_t.begin(), next_t.end(), std::greater<std::pair<double, size_t>>());
                }
            }
        }

        double final_numerator = 0.0;

        for (size_t i = 0; i < code_dim_; ++i) {
            const float signed_o = rotated_unit_data[i];
            const double abs_o = static_cast<double>(std::abs(signed_o));
            int o_bar = static_cast<int>(best_t * abs_o + eps);
            o_bar = std::min<int>(kBaseMax, std::max(0, o_bar));
            final_numerator += (o_bar + 0.5) * abs_o;

            if (signed_o > 0.0f) {
                setSignBit(sign_code, i);
            } else {
                o_bar = static_cast<int>(kBaseMax) - o_bar;
            }

            if ((i & 1U) == 0) {
                compact_code[i / 2] = static_cast<uint8_t>(o_bar & 0x0F);
            } else {
                compact_code[i / 2] |= static_cast<uint8_t>((o_bar & 0x0F) << 4);
            }
        }

        if (final_numerator <= eps || !std::isfinite(final_numerator)) {
            return;
        }

        header.factor_dq = static_cast<float>(2.0 * residual_norm / final_numerator);
    }

    float exrabitqInnerProduct(const QueryContext &query, const void *encoded) const {
        const uint8_t *code = codeBytes(encoded);
        const uint8_t *sign_code = signBytes(encoded);
#if defined(__AVX512F__) || defined(__AVX2__)
        return exrabitqInnerProductSimd(query.rotated.data(), code, sign_code);
#else
        return exrabitqInnerProductScalar(query.rotated.data(), code, sign_code);
#endif
    }

    float exrabitqInnerProductScalar(
        const float *query_rotated,
        const uint8_t *code,
        const uint8_t *sign_code) const {
        float result = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            const float stored = static_cast<float>(compactCodeValue(code, i));
            const float signed_code = signBit(sign_code, i)
                                          ? stored + 0.5f
                                          : stored - kExtendedBias;
            result += query_rotated[i] * signed_code;
        }
        return result;
    }

#if defined(__AVX512F__) || defined(__AVX2__)
    float exrabitqInnerProductSimd(
        const float *query_rotated,
        const uint8_t *code,
        const uint8_t *sign_code) const {
        alignas(64) float signed_code_buffer[32];
        size_t i = 0;
#if defined(__AVX512F__)
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        for (; i + 32 <= code_dim_; i += 32) {
            unpackSignedCodes(code, sign_code, i, 32, signed_code_buffer);
            const __m512 q0 = _mm512_loadu_ps(query_rotated + i);
            const __m512 q1 = _mm512_loadu_ps(query_rotated + i + 16);
            const __m512 c0 = _mm512_load_ps(signed_code_buffer);
            const __m512 c1 = _mm512_load_ps(signed_code_buffer + 16);
            sum0 = _mm512_fmadd_ps(q0, c0, sum0);
            sum1 = _mm512_fmadd_ps(q1, c1, sum1);
        }
        float result = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
#else
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        for (; i + 32 <= code_dim_; i += 32) {
            unpackSignedCodes(code, sign_code, i, 32, signed_code_buffer);
            const __m256 q0 = _mm256_loadu_ps(query_rotated + i);
            const __m256 q1 = _mm256_loadu_ps(query_rotated + i + 8);
            const __m256 q2 = _mm256_loadu_ps(query_rotated + i + 16);
            const __m256 q3 = _mm256_loadu_ps(query_rotated + i + 24);
            const __m256 c0 = _mm256_load_ps(signed_code_buffer);
            const __m256 c1 = _mm256_load_ps(signed_code_buffer + 8);
            const __m256 c2 = _mm256_load_ps(signed_code_buffer + 16);
            const __m256 c3 = _mm256_load_ps(signed_code_buffer + 24);
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q0, c0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q1, c1));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(q2, c2));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(q3, c3));
        }
        __m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, total);
        float result = 0.0f;
        for (float lane : lanes) {
            result += lane;
        }
#endif
        for (; i < code_dim_; ++i) {
            const float stored = static_cast<float>(compactCodeValue(code, i));
            const float signed_code = signBit(sign_code, i)
                                          ? stored + 0.5f
                                          : stored - kExtendedBias;
            result += query_rotated[i] * signed_code;
        }
        return result;
    }

    void unpackSignedCodes(
        const uint8_t *code,
        const uint8_t *sign_code,
        size_t offset,
        size_t count,
        float *signed_code_out) const {
        for (size_t j = 0; j < count; ++j) {
            const size_t dim = offset + j;
            const float stored = static_cast<float>(compactCodeValue(code, dim));
            signed_code_out[j] = signBit(sign_code, dim)
                                     ? stored + 0.5f
                                     : stored - kExtendedBias;
        }
    }
#endif

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
        const EncodedHeader header = loadHeader(encoded);
        return header.norm_sqr + query.query_norm_sqr -
               header.factor_dq * query.query_norm * exrabitqInnerProduct(query, encoded);
    }

    float rawL2Distance(const uint8_t *query, const uint8_t *base) const {
#if defined(__AVX2__)
        const __m256i zero = _mm256_setzero_si256();
        __m256i acc = _mm256_setzero_si256();
        size_t i = 0;
        for (; i + 32 <= dim_; i += 32) {
            const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + i));
            const __m256i b8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base + i));

            const __m256i q_lo = _mm256_unpacklo_epi8(q8, zero);
            const __m256i q_hi = _mm256_unpackhi_epi8(q8, zero);
            const __m256i b_lo = _mm256_unpacklo_epi8(b8, zero);
            const __m256i b_hi = _mm256_unpackhi_epi8(b8, zero);

            const __m256i d_lo = _mm256_sub_epi16(q_lo, b_lo);
            const __m256i d_hi = _mm256_sub_epi16(q_hi, b_hi);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_lo, d_lo));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_hi, d_hi));
        }

        alignas(32) uint32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), acc);
        uint32_t total = 0;
        for (uint32_t lane : lanes) {
            total += lane;
        }
        for (; i < dim_; ++i) {
            const int diff = static_cast<int>(query[i]) - static_cast<int>(base[i]);
            total += static_cast<uint32_t>(diff * diff);
        }
        return static_cast<float>(total);
#else
        uint32_t total = 0;
        for (size_t i = 0; i < dim_; ++i) {
            const int diff = static_cast<int>(query[i]) - static_cast<int>(base[i]);
            total += static_cast<uint32_t>(diff * diff);
        }
        return static_cast<float>(total);
#endif
    }

    void batchAccumulateScalar(
        const QueryContext &query,
        const void *const *data_points,
        size_t count,
        uint32_t *accumulations) const {
        for (size_t idx = 0; idx < count; ++idx) {
            accumulations[idx] = scalarCodeAccumulation(query, data_points[idx]);
        }
    }

#if defined(__AVX512F__)
    void batchAccumulateAvx512(
        const QueryContext &query,
        const void *const *data_points,
        size_t count,
        uint32_t *accumulations) const {
        size_t idx = 0;
        for (; idx + 16 <= count; idx += 16) {
            __m512i acc = _mm512_setzero_si512();
            for (size_t dim = 0; dim < code_dim_; dim += 2) {
                alignas(64) uint32_t lo_values[16];
                alignas(64) uint32_t hi_values[16];
                for (size_t lane = 0; lane < 16; ++lane) {
                    const uint8_t byte = codeBytes(data_points[idx + lane])[dim / 2];
                    lo_values[lane] = static_cast<uint32_t>(byte & 0x0F);
                    hi_values[lane] = static_cast<uint32_t>(byte >> 4);
                }
                __m512i lo = _mm512_load_si512(reinterpret_cast<const __m512i *>(lo_values));
                __m512i hi = _mm512_load_si512(reinterpret_cast<const __m512i *>(hi_values));
                lo = _mm512_mullo_epi32(lo, _mm512_set1_epi32(query.byte_query[dim]));
                hi = _mm512_mullo_epi32(hi, _mm512_set1_epi32(query.byte_query[dim + 1]));
                acc = _mm512_add_epi32(acc, lo);
                acc = _mm512_add_epi32(acc, hi);
            }
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&accumulations[idx]), acc);
        }
        if (idx < count) {
            batchAccumulateScalar(query, data_points + idx, count - idx, accumulations + idx);
        }
    }
#elif defined(__AVX2__)
    void batchAccumulateAvx2(
        const QueryContext &query,
        const void *const *data_points,
        size_t count,
        uint32_t *accumulations) const {
        size_t idx = 0;
        for (; idx + 8 <= count; idx += 8) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t dim = 0; dim < code_dim_; dim += 2) {
                alignas(32) uint32_t lo_values[8];
                alignas(32) uint32_t hi_values[8];
                for (size_t lane = 0; lane < 8; ++lane) {
                    const uint8_t byte = codeBytes(data_points[idx + lane])[dim / 2];
                    lo_values[lane] = static_cast<uint32_t>(byte & 0x0F);
                    hi_values[lane] = static_cast<uint32_t>(byte >> 4);
                }
                __m256i lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(lo_values));
                __m256i hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(hi_values));
                lo = _mm256_mullo_epi32(lo, _mm256_set1_epi32(query.byte_query[dim]));
                hi = _mm256_mullo_epi32(hi, _mm256_set1_epi32(query.byte_query[dim + 1]));
                acc = _mm256_add_epi32(acc, lo);
                acc = _mm256_add_epi32(acc, hi);
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&accumulations[idx]), acc);
        }
        if (idx < count) {
            batchAccumulateScalar(query, data_points + idx, count - idx, accumulations + idx);
        }
    }
#endif

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
        const uint8_t *sign_code = signBytes(encoded);
        decoded.assign(code_dim_, 0.0f);

        float code_norm_sqr = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            const float stored = static_cast<float>(compactCodeValue(code, i));
            const float signed_code = signBit(sign_code, i)
                                          ? stored + 0.5f
                                          : stored - kExtendedBias;
            code_norm_sqr += signed_code * signed_code;
        }
        const float scale = code_norm_sqr > 0.0f
                                ? std::sqrt(header.norm_sqr / code_norm_sqr)
                                : 0.0f;

        for (size_t i = 0; i < code_dim_; ++i) {
            const float stored = static_cast<float>(compactCodeValue(code, i));
            const float signed_code = signBit(sign_code, i)
                                          ? stored + 0.5f
                                          : stored - kExtendedBias;
            decoded[i] = rotated_global_center_[i] + scale * signed_code;
        }
    }

 public:
    explicit RaBitQSpace(size_t dim, size_t centroid_count = 1, uint32_t random_seed = 100)
        : dim_(dim),
          code_dim_(roundUp64(dim)),
          compact_code_bytes_(code_dim_ / 2),
          sign_code_bytes_(code_dim_ / 8),
          data_size_(sizeof(EncodedHeader) + compact_code_bytes_ + sign_code_bytes_),
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
        const std::string magic = "EXRBTQ06";
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
        if (!input.good() || std::string(magic, sizeof(magic)) != "EXRBTQ06") {
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
        fht_signs_.assign(code_dim_, 0.0f);
        input.read(reinterpret_cast<char *>(&random_seed_), sizeof(random_seed_));
        input.read(reinterpret_cast<char *>(global_center_.data()), global_center_.size() * sizeof(float));
        input.read(reinterpret_cast<char *>(fht_signs_.data()), fht_signs_.size() * sizeof(float));

        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read ExRaBitQ state payload");
        }
        refreshRotatedCenter();
    }

    void saveRawStore(std::ostream &output) const {
        std::lock_guard<std::mutex> lock(raw_store_mutex_);
        const std::string magic = "EXRRAW01";
        output.write(magic.data(), magic.size());

        const uint64_t dim = static_cast<uint64_t>(dim_);
        const uint64_t raw_count = dim_ == 0 ? 0 : static_cast<uint64_t>(raw_store_.size() / dim_);
        output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        output.write(reinterpret_cast<const char *>(&raw_count), sizeof(raw_count));
        if (!raw_store_.empty()) {
            output.write(reinterpret_cast<const char *>(raw_store_.data()), raw_store_.size());
        }
        if (!output.good()) {
            throw std::runtime_error("RaBitQSpace failed to save raw rerank store");
        }
    }

    void loadRawStore(std::istream &input) {
        char magic[8];
        input.read(magic, sizeof(magic));
        if (!input.good() || std::string(magic, sizeof(magic)) != "EXRRAW01") {
            throw std::runtime_error("RaBitQSpace invalid raw rerank store; rebuild the index");
        }

        uint64_t stored_dim = 0;
        uint64_t raw_count = 0;
        input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        input.read(reinterpret_cast<char *>(&raw_count), sizeof(raw_count));
        if (!input.good() || stored_dim != dim_) {
            throw std::runtime_error("RaBitQSpace raw rerank store is incompatible with this index");
        }

        std::lock_guard<std::mutex> lock(raw_store_mutex_);
        raw_store_.assign(static_cast<size_t>(raw_count) * dim_, 0);
        if (!raw_store_.empty()) {
            input.read(reinterpret_cast<char *>(raw_store_.data()), raw_store_.size());
        }
        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read raw rerank store payload");
        }
        pendingRawStorage().clear();
    }

    size_t rawStoreCount() const {
        std::lock_guard<std::mutex> lock(raw_store_mutex_);
        return dim_ == 0 ? 0 : raw_store_.size() / dim_;
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
        query->byte_query.assign(code_dim_, 0);
        query->raw_query_u8.assign(dim_, 0);

        query->query_norm_sqr = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            query->residual[i] = raw_query[i] - global_center_[i];
            query->query_norm_sqr += query->residual[i] * query->residual[i];
            int raw_value = static_cast<int>(std::lround(raw_query[i]));
            raw_value = std::min(255, std::max(0, raw_value));
            query->raw_query_u8[i] = static_cast<uint8_t>(raw_value);
        }

        query->query_norm = std::sqrt(query->query_norm_sqr);
        if (query->query_norm > 0.0f) {
            const float inv_norm = 1.0f / query->query_norm;
            for (float &value : query->residual) {
                value *= inv_norm;
            }
        }

        rotate(query->residual.data(), query->rotated);

        float lower_val = std::numeric_limits<float>::max();
        float upper_val = std::numeric_limits<float>::lowest();
        for (float value : query->rotated) {
            lower_val = std::min(lower_val, value);
            upper_val = std::max(upper_val, value);
        }
        query->lower_val = lower_val;
        query->width = (upper_val - lower_val) / static_cast<float>((1 << kQueryBits) - 1);
        if (query->width <= 0.0f) {
            query->lower_val = 0.0f;
            query->width = 1.0f;
        }

        const float inv_width = 1.0f / query->width;
        int32_t sumq = 0;
        for (size_t i = 0; i < code_dim_; ++i) {
            int value = static_cast<int>(std::lround((query->rotated[i] - query->lower_val) * inv_width));
            value = std::min((1 << kQueryBits) - 1, std::max(0, value));
            query->byte_query[i] = static_cast<uint8_t>(value);
            sumq += value;
        }
        query->sumq = sumq;

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

    float result_distance_by_id(const void *prepared_query, size_t internal_id, const void *data_point) override {
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        const uint8_t *raw = rawByInternalId(internal_id);
        if (raw == nullptr) {
            return query_distance(prepared_query, data_point);
        }
        return rawL2Distance(query.raw_query_u8.data(), raw);
    }

    void prepare_data_for_add(const void *raw_data_point) override {
        const float *raw_vector = static_cast<const float *>(raw_data_point);
        std::vector<uint8_t> &pending_raw = pendingRawStorage();
        pending_raw.assign(dim_, 0);
        encodeRawVector(raw_vector, pending_raw.data());
    }

    void commit_data_for_add(size_t internal_id, const void *data_point) override {
        (void) data_point;
        std::vector<uint8_t> &pending_raw = pendingRawStorage();
        if (pending_raw.size() != dim_) {
            return;
        }
        std::lock_guard<std::mutex> lock(raw_store_mutex_);
        const size_t required = (internal_id + 1) * dim_;
        if (raw_store_.size() < required) {
            raw_store_.resize(required, 0);
        }
        std::memcpy(raw_store_.data() + internal_id * dim_, pending_raw.data(), dim_);
        pending_raw.clear();
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
        uint8_t *sign_code = signBytes(encoded_out);
        exrabitqGlobalCode(rotated_unit.data(), residual_norm, code, sign_code, header);
        std::memcpy(encoded_out, &header, sizeof(header));
    }

    std::vector<char> encodeVector(const float *raw_vector) const {
        std::vector<char> encoded(data_size_, 0);
        encodeVector(raw_vector, encoded.data());
        return encoded;
    }
};

}  // namespace hnswlib
