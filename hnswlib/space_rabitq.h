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

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hnswlib.h"

namespace hnswlib {

class RaBitQSpace : public SpaceInterface<float> {
 public:
    static constexpr size_t kTotalBits = 4;
    static constexpr size_t kRemainingBits = 3;
    static constexpr uint32_t kUnsignedMax = 15;
    static constexpr uint32_t kRemainingMax = 7;
    static constexpr uint32_t kMsbWeight = 8;
    static constexpr float kUnsignedOffset = 7.5f;
    static constexpr float kRemainingOffset = 3.5f;
    static constexpr size_t kResidualBlockSize = 16;
    static constexpr double kExtendedRaBitQErrorConstant = 5.75;

    struct EncodedHeader {
        float norm_sqr;
        float long_scale;
    };

    struct ShortCodeFactors {
        float short_scale;
        float error_scale;
    };

    struct ResidualCodeFactors {
        float residual_norm_sqr;
        float long_residual_inner_product;
    };

    struct QueryContext {
        std::vector<float> rotated_residual;
        std::vector<float> rotated_residual_even;
        std::vector<float> rotated_residual_odd;
        mutable std::vector<float> residual_nibble_lut;
        float query_norm = 0.0f;
        float query_norm_sqr = 0.0f;
        float half_sum_residual = 0.0f;
        float rotated_residual_sum = 0.0f;
        mutable bool residual_nibble_lut_ready = false;
    };

    struct LongCodeIps {
        float short_ip;
        float remaining_ip;
    };

 private:
    size_t dim_{0};
    size_t code_dim_{0};
    size_t compact_code_bytes_{0};
    size_t short_factor_bytes_{0};
    size_t residual_factor_bytes_{0};
    size_t residual_block_count_{0};
    size_t residual_scale_bytes_{0};
    size_t residual_code_bytes_{0};
    size_t full_data_size_{0};
    size_t residual_disk_record_bytes_{0};
    size_t data_size_{0};
    float inv_sqrt_code_dim_{1.0f};
    bool external_residual_storage_{false};
    mutable int residual_fd_{-1};
    mutable const char *residual_mmap_{nullptr};
    mutable size_t residual_mmap_bytes_{0};

    DISTFUNC<float> fstdistfunc_{nullptr};

    uint32_t random_seed_{100};
    std::vector<float> fht_signs_;
    std::vector<float> global_center_;

    static size_t roundUp64(size_t value) {
        const size_t rounded = ((value + 63U) / 64U) * 64U;
        size_t power = 1;
        while (power < rounded) {
            power <<= 1U;
        }
        return power;
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
        const size_t offset = external_residual_storage_
            ? sizeof(EncodedHeader) + sizeof(ShortCodeFactors)
            : sizeof(EncodedHeader) + sizeof(ShortCodeFactors) + sizeof(ResidualCodeFactors) +
                residual_scale_bytes_;
        return reinterpret_cast<const uint8_t *>(
            static_cast<const char *>(encoded) + offset);
    }

    uint8_t *codeBytes(void *encoded) const {
        const size_t offset = external_residual_storage_
            ? sizeof(EncodedHeader) + sizeof(ShortCodeFactors)
            : sizeof(EncodedHeader) + sizeof(ShortCodeFactors) + sizeof(ResidualCodeFactors) +
                residual_scale_bytes_;
        return reinterpret_cast<uint8_t *>(
            static_cast<char *>(encoded) + offset);
    }

    const uint8_t *codeBytesFull(const void *encoded) const {
        return reinterpret_cast<const uint8_t *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors) +
                sizeof(ResidualCodeFactors) + residual_scale_bytes_);
    }

    uint8_t *codeBytesFull(void *encoded) const {
        return reinterpret_cast<uint8_t *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors) +
                sizeof(ResidualCodeFactors) + residual_scale_bytes_);
    }

    const uint8_t *residualCodeBytes(const void *encoded) const {
        return reinterpret_cast<const uint8_t *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors) +
                sizeof(ResidualCodeFactors) + residual_scale_bytes_ + compact_code_bytes_);
    }

    uint8_t *residualCodeBytes(void *encoded) const {
        return reinterpret_cast<uint8_t *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors) +
                sizeof(ResidualCodeFactors) + residual_scale_bytes_ + compact_code_bytes_);
    }

    const float *residualScales(const void *encoded) const {
        return reinterpret_cast<const float *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors) +
                sizeof(ResidualCodeFactors));
    }

    float *residualScales(void *encoded) const {
        return reinterpret_cast<float *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors) +
                sizeof(ResidualCodeFactors));
    }

    static uint8_t codeValue(const uint8_t *packed_code, size_t index) {
        const uint8_t byte = packed_code[index >> 1U];
        return static_cast<uint8_t>((index & 1U) ? (byte >> 4U) : (byte & 0x0FU));
    }

    static void setCodeValue(uint8_t *packed_code, size_t index, uint8_t value) {
        value = static_cast<uint8_t>(value & 0x0FU);
        uint8_t &byte = packed_code[index >> 1U];
        if (index & 1U) {
            byte = static_cast<uint8_t>((byte & 0x0FU) | (value << 4U));
        } else {
            byte = static_cast<uint8_t>((byte & 0xF0U) | value);
        }
    }

    const ShortCodeFactors *shortFactors(const void *encoded) const {
        return reinterpret_cast<const ShortCodeFactors *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader));
    }

    ShortCodeFactors *shortFactors(void *encoded) const {
        return reinterpret_cast<ShortCodeFactors *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader));
    }

    const ResidualCodeFactors *residualFactors(const void *encoded) const {
        return reinterpret_cast<const ResidualCodeFactors *>(
            static_cast<const char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors));
    }

    ResidualCodeFactors *residualFactors(void *encoded) const {
        return reinterpret_cast<ResidualCodeFactors *>(
            static_cast<char *>(encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors));
    }

    const ResidualCodeFactors *residualFactorsFromRecord(const void *record) const {
        return reinterpret_cast<const ResidualCodeFactors *>(record);
    }

    const float *residualScalesFromRecord(const void *record) const {
        return reinterpret_cast<const float *>(
            static_cast<const char *>(record) + sizeof(ResidualCodeFactors));
    }

    const uint8_t *residualCodeBytesFromRecord(const void *record) const {
        return reinterpret_cast<const uint8_t *>(
            static_cast<const char *>(record) + sizeof(ResidualCodeFactors) + residual_scale_bytes_);
    }

    const char *externalResidualRecord(size_t id) const {
        if (residual_mmap_ == nullptr) {
            throw std::runtime_error("RaBitQ external residual storage is not open");
        }
        const size_t offset = id * residual_disk_record_bytes_;
        if (offset + residual_disk_record_bytes_ > residual_mmap_bytes_) {
            throw std::runtime_error("RaBitQ external residual id is outside residual file");
        }
        return residual_mmap_ + offset;
    }

    void fastQuantizeAbs(const float *abs_unit_data, uint8_t *abs_code, float &ip_norm) const {
        constexpr double eps = 1e-5;
        constexpr int n_enum = 10;
        double max_o = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            max_o = std::max(max_o, static_cast<double>(abs_unit_data[i]));
        }
        if (max_o <= eps) {
            std::memset(abs_code, 0, code_dim_);
            ip_norm = 1.0f;
            return;
        }

        const double t_start = static_cast<double>((kRemainingMax / 3U)) / max_o;
        const double t_end = (static_cast<double>(kRemainingMax) + n_enum) / max_o;
        thread_local std::vector<int> cur_code;
        cur_code.assign(code_dim_, 0);

        double sqr_denominator = static_cast<double>(code_dim_) * 0.25;
        double numerator = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            cur_code[i] = static_cast<int>(t_start * abs_unit_data[i] + eps);
            cur_code[i] = std::min<int>(cur_code[i], static_cast<int>(kRemainingMax));
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

            if (update_code < static_cast<int>(kRemainingMax)) {
                const double candidate_t = static_cast<double>(update_code + 1) / abs_unit_data[update_id];
                if (candidate_t < t_end) {
                    next_t.emplace(candidate_t, update_id);
                }
            }
        }

        numerator = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            int value = static_cast<int>(best_t * abs_unit_data[i] + eps);
            value = std::min<int>(value, static_cast<int>(kRemainingMax));
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

    float dotRemainingUint8FloatAvxDispatch(const uint8_t *remaining_code, const QueryContext &query) const {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        return dotRemainingUint8FloatAvx512(remaining_code, query);
#elif defined(__AVX2__)
        return dotRemainingUint8FloatAvx2(remaining_code, query);
#else
        return dotRemainingUint8FloatScalar(remaining_code, query);
#endif
    }

    float dotRemainingUint8FloatScalar(const uint8_t *remaining_code, const QueryContext &query) const {
        float result = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            result += static_cast<float>(codeValue(remaining_code, i) & kRemainingMax) * query.rotated_residual[i];
        }
        return result;
    }

#if defined(__AVX2__)
    float dotRemainingUint8FloatAvx2(const uint8_t *remaining_code, const QueryContext &query) const {
        __m256 sum = _mm256_setzero_ps();
        const __m128i low_mask = _mm_set1_epi8(static_cast<char>(kRemainingMax));
        const size_t pair_count = code_dim_ >> 1U;
        size_t pair = 0;
        for (; pair + 8 <= pair_count; pair += 8) {
            const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(remaining_code + pair));
            const __m128i lo = _mm_and_si128(packed, low_mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
            const __m256 lo_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
            const __m256 hi_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(lo_f, _mm256_loadu_ps(query.rotated_residual_even.data() + pair)));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(hi_f, _mm256_loadu_ps(query.rotated_residual_odd.data() + pair)));
        }

        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sum);
        float result = 0.0f;
        for (float lane : lanes) {
            result += lane;
        }
        for (; pair < pair_count; ++pair) {
            const uint8_t packed = remaining_code[pair];
            result += static_cast<float>(packed & kRemainingMax) * query.rotated_residual_even[pair];
            result += static_cast<float>((packed >> 4U) & kRemainingMax) * query.rotated_residual_odd[pair];
        }
        return result;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    float dotRemainingUint8FloatAvx512(const uint8_t *remaining_code, const QueryContext &query) const {
        __m512 sum = _mm512_setzero_ps();
        const __m128i low_mask = _mm_set1_epi8(static_cast<char>(kRemainingMax));
        const size_t pair_count = code_dim_ >> 1U;
        size_t pair = 0;
        for (; pair + 16 <= pair_count; pair += 16) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(remaining_code + pair));
            const __m128i lo = _mm_and_si128(packed, low_mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
            const __m512 lo_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo));
            const __m512 hi_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(lo_f, _mm512_loadu_ps(query.rotated_residual_even.data() + pair)));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(hi_f, _mm512_loadu_ps(query.rotated_residual_odd.data() + pair)));
        }

        float result = _mm512_reduce_add_ps(sum);
        for (; pair < pair_count; ++pair) {
            const uint8_t packed = remaining_code[pair];
            result += static_cast<float>(packed & kRemainingMax) * query.rotated_residual_even[pair];
            result += static_cast<float>((packed >> 4U) & kRemainingMax) * query.rotated_residual_odd[pair];
        }
        return result;
    }
#endif

    float shortCodeIpFromFullCodeAvxDispatch(const QueryContext &query, const uint8_t *code) const {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        return shortCodeIpFromFullCodeAvx512(query, code);
#elif defined(__AVX2__)
        return shortCodeIpFromFullCodeAvx2(query, code);
#else
        return shortCodeIpFromFullCodeScalar(query, code);
#endif
    }

    float shortCodeIpFromFullCodeScalar(const QueryContext &query, const uint8_t *code) const {
        float selected_sum = 0.0f;
        for (size_t i = 0; i < code_dim_; ++i) {
            selected_sum += (codeValue(code, i) >> kRemainingBits) ? query.rotated_residual[i] : 0.0f;
        }
        return selected_sum - query.half_sum_residual;
    }

#if defined(__AVX2__)
    float shortCodeIpFromFullCodeAvx2(const QueryContext &query, const uint8_t *code) const {
        __m256 sum = _mm256_setzero_ps();
        const __m128i one_mask = _mm_set1_epi8(1);
        const size_t pair_count = code_dim_ >> 1U;
        size_t pair = 0;
        for (; pair + 8 <= pair_count; pair += 8) {
            const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(code + pair));
            const __m128i lo_bit = _mm_and_si128(_mm_srli_epi16(packed, kRemainingBits), one_mask);
            const __m128i hi_bit = _mm_and_si128(_mm_srli_epi16(packed, kTotalBits + kRemainingBits), one_mask);
            const __m256 lo_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_bit));
            const __m256 hi_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_bit));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(lo_f, _mm256_loadu_ps(query.rotated_residual_even.data() + pair)));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(hi_f, _mm256_loadu_ps(query.rotated_residual_odd.data() + pair)));
        }

        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sum);
        float selected_sum = 0.0f;
        for (float lane : lanes) {
            selected_sum += lane;
        }
        for (; pair < pair_count; ++pair) {
            const uint8_t packed = code[pair];
            selected_sum += ((packed >> kRemainingBits) & 1U) ? query.rotated_residual_even[pair] : 0.0f;
            selected_sum += ((packed >> (kTotalBits + kRemainingBits)) & 1U) ? query.rotated_residual_odd[pair] : 0.0f;
        }
        return selected_sum - query.half_sum_residual;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    float shortCodeIpFromFullCodeAvx512(const QueryContext &query, const uint8_t *code) const {
        __m512 sum = _mm512_setzero_ps();
        const __m128i one_mask = _mm_set1_epi8(1);
        const size_t pair_count = code_dim_ >> 1U;
        size_t pair = 0;
        for (; pair + 16 <= pair_count; pair += 16) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(code + pair));
            const __m128i lo_bit = _mm_and_si128(_mm_srli_epi16(packed, kRemainingBits), one_mask);
            const __m128i hi_bit = _mm_and_si128(_mm_srli_epi16(packed, kTotalBits + kRemainingBits), one_mask);
            const __m512 lo_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo_bit));
            const __m512 hi_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi_bit));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(lo_f, _mm512_loadu_ps(query.rotated_residual_even.data() + pair)));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(hi_f, _mm512_loadu_ps(query.rotated_residual_odd.data() + pair)));
        }

        float selected_sum = _mm512_reduce_add_ps(sum);
        for (; pair < pair_count; ++pair) {
            const uint8_t packed = code[pair];
            selected_sum += ((packed >> kRemainingBits) & 1U) ? query.rotated_residual_even[pair] : 0.0f;
            selected_sum += ((packed >> (kTotalBits + kRemainingBits)) & 1U) ? query.rotated_residual_odd[pair] : 0.0f;
        }
        return selected_sum - query.half_sum_residual;
    }
#endif

    LongCodeIps longCodeIpsDispatch(const QueryContext &query, const uint8_t *code) const {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        return longCodeIpsAvx512(query, code);
#elif defined(__AVX2__)
        return longCodeIpsAvx2(query, code);
#else
        return longCodeIpsScalar(query, code);
#endif
    }

    LongCodeIps longCodeIpsScalar(const QueryContext &query, const uint8_t *code) const {
        float selected_sum = 0.0f;
        float remaining_ip = 0.0f;
        const size_t pair_count = code_dim_ >> 1U;
        for (size_t pair = 0; pair < pair_count; ++pair) {
            const uint8_t packed = code[pair];
            const uint8_t lo = static_cast<uint8_t>(packed & 0x0FU);
            const uint8_t hi = static_cast<uint8_t>(packed >> 4U);
            selected_sum += (lo >> kRemainingBits) ? query.rotated_residual_even[pair] : 0.0f;
            selected_sum += (hi >> kRemainingBits) ? query.rotated_residual_odd[pair] : 0.0f;
            remaining_ip += static_cast<float>(lo & kRemainingMax) * query.rotated_residual_even[pair];
            remaining_ip += static_cast<float>(hi & kRemainingMax) * query.rotated_residual_odd[pair];
        }
        return LongCodeIps{selected_sum - query.half_sum_residual, remaining_ip};
    }

#if defined(__AVX2__)
    LongCodeIps longCodeIpsAvx2(const QueryContext &query, const uint8_t *code) const {
        __m256 short_sum = _mm256_setzero_ps();
        __m256 remaining_sum = _mm256_setzero_ps();
        const __m128i remaining_mask = _mm_set1_epi8(static_cast<char>(kRemainingMax));
        const __m128i one_mask = _mm_set1_epi8(1);
        const size_t pair_count = code_dim_ >> 1U;
        size_t pair = 0;
        for (; pair + 8 <= pair_count; pair += 8) {
            const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(code + pair));
            const __m128i lo = _mm_and_si128(packed, remaining_mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), remaining_mask);
            const __m128i lo_bit = _mm_and_si128(_mm_srli_epi16(packed, kRemainingBits), one_mask);
            const __m128i hi_bit = _mm_and_si128(_mm_srli_epi16(packed, kTotalBits + kRemainingBits), one_mask);
            const __m256 even_q = _mm256_loadu_ps(query.rotated_residual_even.data() + pair);
            const __m256 odd_q = _mm256_loadu_ps(query.rotated_residual_odd.data() + pair);
            remaining_sum = _mm256_add_ps(
                remaining_sum,
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), even_q));
            remaining_sum = _mm256_add_ps(
                remaining_sum,
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), odd_q));
            short_sum = _mm256_add_ps(
                short_sum,
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_bit)), even_q));
            short_sum = _mm256_add_ps(
                short_sum,
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi_bit)), odd_q));
        }

        alignas(32) float short_lanes[8];
        alignas(32) float remaining_lanes[8];
        _mm256_store_ps(short_lanes, short_sum);
        _mm256_store_ps(remaining_lanes, remaining_sum);
        float selected_sum = 0.0f;
        float remaining_ip = 0.0f;
        for (size_t lane = 0; lane < 8; ++lane) {
            selected_sum += short_lanes[lane];
            remaining_ip += remaining_lanes[lane];
        }
        for (; pair < pair_count; ++pair) {
            const uint8_t packed = code[pair];
            const uint8_t lo = static_cast<uint8_t>(packed & 0x0FU);
            const uint8_t hi = static_cast<uint8_t>(packed >> 4U);
            selected_sum += (lo >> kRemainingBits) ? query.rotated_residual_even[pair] : 0.0f;
            selected_sum += (hi >> kRemainingBits) ? query.rotated_residual_odd[pair] : 0.0f;
            remaining_ip += static_cast<float>(lo & kRemainingMax) * query.rotated_residual_even[pair];
            remaining_ip += static_cast<float>(hi & kRemainingMax) * query.rotated_residual_odd[pair];
        }
        return LongCodeIps{selected_sum - query.half_sum_residual, remaining_ip};
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    LongCodeIps longCodeIpsAvx512(const QueryContext &query, const uint8_t *code) const {
        __m512 short_sum = _mm512_setzero_ps();
        __m512 remaining_sum = _mm512_setzero_ps();
        const __m128i remaining_mask = _mm_set1_epi8(static_cast<char>(kRemainingMax));
        const __m128i one_mask = _mm_set1_epi8(1);
        const size_t pair_count = code_dim_ >> 1U;
        size_t pair = 0;
        for (; pair + 16 <= pair_count; pair += 16) {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(code + pair));
            const __m128i lo = _mm_and_si128(packed, remaining_mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), remaining_mask);
            const __m128i lo_bit = _mm_and_si128(_mm_srli_epi16(packed, kRemainingBits), one_mask);
            const __m128i hi_bit = _mm_and_si128(_mm_srli_epi16(packed, kTotalBits + kRemainingBits), one_mask);
            const __m512 even_q = _mm512_loadu_ps(query.rotated_residual_even.data() + pair);
            const __m512 odd_q = _mm512_loadu_ps(query.rotated_residual_odd.data() + pair);
            remaining_sum = _mm512_add_ps(
                remaining_sum,
                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo)), even_q));
            remaining_sum = _mm512_add_ps(
                remaining_sum,
                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi)), odd_q));
            short_sum = _mm512_add_ps(
                short_sum,
                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo_bit)), even_q));
            short_sum = _mm512_add_ps(
                short_sum,
                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi_bit)), odd_q));
        }

        float selected_sum = _mm512_reduce_add_ps(short_sum);
        float remaining_ip = _mm512_reduce_add_ps(remaining_sum);
        for (; pair < pair_count; ++pair) {
            const uint8_t packed = code[pair];
            const uint8_t lo = static_cast<uint8_t>(packed & 0x0FU);
            const uint8_t hi = static_cast<uint8_t>(packed >> 4U);
            selected_sum += (lo >> kRemainingBits) ? query.rotated_residual_even[pair] : 0.0f;
            selected_sum += (hi >> kRemainingBits) ? query.rotated_residual_odd[pair] : 0.0f;
            remaining_ip += static_cast<float>(lo & kRemainingMax) * query.rotated_residual_even[pair];
            remaining_ip += static_cast<float>(hi & kRemainingMax) * query.rotated_residual_odd[pair];
        }
        return LongCodeIps{selected_sum - query.half_sum_residual, remaining_ip};
    }
#endif

    float queryDistanceLong(const QueryContext &query, const void *encoded) const {
        const EncodedHeader header = loadHeader(encoded);
        if (header.long_scale <= 0.0f || !std::isfinite(header.long_scale)) {
            return header.norm_sqr + query.query_norm_sqr;
        }
        const LongCodeIps ips = longCodeIpsDispatch(query, codeBytes(encoded));
        return queryDistanceLongWithIps(query, header, ips.short_ip, ips.remaining_ip);
    }

    float queryDistanceLongWithShortIp(
        const QueryContext &query,
        const void *encoded,
        const EncodedHeader &header,
        float short_ip) const {
        if (header.long_scale <= 0.0f || !std::isfinite(header.long_scale)) {
            return header.norm_sqr + query.query_norm_sqr;
        }
        const float remaining_ip = dotRemainingUint8FloatAvxDispatch(codeBytes(encoded), query);
        return queryDistanceLongWithIps(query, header, short_ip, remaining_ip);
    }

    float queryDistanceLongWithIps(
        const QueryContext &query,
        const EncodedHeader &header,
        float short_ip,
        float remaining_ip) const {
        const float signed_long_ip =
            static_cast<float>(kMsbWeight) * short_ip + remaining_ip -
            static_cast<float>(kRemainingMax) * query.half_sum_residual;
        const float estimated_inner = header.long_scale * signed_long_ip;
        return header.norm_sqr + query.query_norm_sqr -
               estimated_inner;
    }

    float queryDistanceLowerBound(const QueryContext &query, const void *encoded) const {
        const EncodedHeader header = loadHeader(encoded);
        const ShortCodeFactors factors = *shortFactors(encoded);
        const float ip_xb_q = shortCodeIp(query, codeBytes(encoded));
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

    DistanceInterval computeShortDistanceIntervalTyped(const QueryContext &query, const void *encoded) const {
        const EncodedHeader header = loadHeader(encoded);
        const ShortCodeFactors factors = *shortFactors(encoded);
        const float short_ip = shortCodeIp(query, codeBytes(encoded));
        const float compensated_ip = short_ip * factors.short_scale;
        const float err = factors.error_scale * query.query_norm;
        const float estimate = header.norm_sqr + query.query_norm_sqr - compensated_ip;
        return DistanceInterval{
            estimate,
            estimate - err,
            estimate + err};
    }

    DistanceInterval computeLongDistanceIntervalTyped(const QueryContext &query, const void *encoded) const {
        const float distance = queryDistanceLong(query, encoded);
        const ShortCodeFactors factors = *shortFactors(encoded);
        const float err = factors.error_scale * query.query_norm;
        return DistanceInterval{
            distance,
            distance - err,
            distance + err};
    }

    DistanceInterval computeResidualDistanceIntervalTyped(
        const QueryContext &query,
        const void *encoded,
        float long_distance) const {
        if (external_residual_storage_) {
            throw std::runtime_error("single residual distance cannot use external residual storage");
        }
        const ResidualCodeFactors factors = *residualFactors(encoded);
        const float q_residual_ip =
            residualInt8Ip(query, residualCodeBytes(encoded), residualScales(encoded));
        (void) factors;
        const float distance = long_distance - 2.0f * q_residual_ip;
        return DistanceInterval{distance, distance, distance};
    }

    DistanceInterval computeResidualDistanceIntervalFromRecord(
        const QueryContext &query,
        const void *record,
        float long_distance) const {
        const ResidualCodeFactors factors = *residualFactorsFromRecord(record);
        const float q_residual_ip =
            residualInt8Ip(query, residualCodeBytesFromRecord(record), residualScalesFromRecord(record));
        (void) factors;
        const float distance = long_distance - 2.0f * q_residual_ip;
        return DistanceInterval{distance, distance, distance};
    }

    float shortCodeIp(const QueryContext &query, const uint8_t *full_code) const {
        return shortCodeIpFromFullCodeAvxDispatch(query, full_code);
    }

    float residualSignIp(const QueryContext &query, const uint8_t *residual_code) const {
        ensureResidualNibbleLut(query);
        const float *lut = query.residual_nibble_lut.data();
        float positive_sum = 0.0f;
        size_t group = 0;
        for (size_t byte_index = 0; byte_index < residual_code_bytes_; ++byte_index) {
            const uint8_t packed = residual_code[byte_index];
            positive_sum += lut[(group << 4U) + (packed & 0x0FU)];
            ++group;
            if (group < ((code_dim_ + 3U) >> 2U)) {
                positive_sum += lut[(group << 4U) + (packed >> 4U)];
                ++group;
            }
        }
        return 2.0f * positive_sum - query.rotated_residual_sum;
    }

    float residualInt8Ip(
        const QueryContext &query,
        const uint8_t *residual_code,
        const float *residual_scales) const {
        const int8_t *codes = reinterpret_cast<const int8_t *>(residual_code);
        float result = 0.0f;
        for (size_t block = 0; block < residual_block_count_; ++block) {
            const float scale = residual_scales[block];
            if (scale == 0.0f || !std::isfinite(scale)) {
                continue;
            }
            const size_t begin = block * kResidualBlockSize;
            const size_t end = std::min(code_dim_, begin + kResidualBlockSize);
            float block_ip = 0.0f;
            for (size_t i = begin; i < end; ++i) {
                block_ip += static_cast<float>(codes[i]) * query.rotated_residual[i];
            }
            result += scale * block_ip;
        }
        return result;
    }

    void ensureResidualNibbleLut(const QueryContext &query) const {
        if (query.residual_nibble_lut_ready) {
            return;
        }
        const size_t group_count = (code_dim_ + 3U) >> 2U;
        query.residual_nibble_lut.assign(group_count * 16U, 0.0f);
        for (size_t group = 0; group < group_count; ++group) {
            const size_t base_dim = group << 2U;
            float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t lane = 0; lane < 4U; ++lane) {
                const size_t dim = base_dim + lane;
                if (dim < code_dim_) {
                    values[lane] = query.rotated_residual[dim];
                }
            }
            float *group_lut = query.residual_nibble_lut.data() + (group << 4U);
            for (uint8_t mask = 1; mask < 16U; ++mask) {
                const uint8_t lsb = static_cast<uint8_t>(mask & static_cast<uint8_t>(-mask));
                const uint8_t lane =
                    lsb == 1U ? 0U :
                    lsb == 2U ? 1U :
                    lsb == 4U ? 2U : 3U;
                group_lut[mask] = group_lut[mask ^ lsb] + values[lane];
            }
        }
        query.residual_nibble_lut_ready = true;
    }

    static void setResidualSignBit(uint8_t *residual_code, size_t index, bool positive) {
        const uint8_t mask = static_cast<uint8_t>(1U << (index & 7U));
        if (positive) {
            residual_code[index >> 3U] = static_cast<uint8_t>(residual_code[index >> 3U] | mask);
        } else {
            residual_code[index >> 3U] = static_cast<uint8_t>(residual_code[index >> 3U] & ~mask);
        }
    }

    void batchShortCodeIp(
        const QueryContext &query,
        const void *const *data_points,
        size_t count,
        float *short_ips) const {
        for (size_t i = 0; i < count; ++i) {
            short_ips[i] = shortCodeIp(query, codeBytes(data_points[i]));
        }
    }

    float distanceBetweenEncoded(const char *lhs, const char *rhs) const {
        const EncodedHeader lhs_header = loadHeader(lhs);
        const EncodedHeader rhs_header = loadHeader(rhs);
        const uint8_t *lhs_code = codeBytes(lhs);
        const uint8_t *rhs_code = codeBytes(rhs);

        double code_ip = 0.0;
        for (size_t i = 0; i < code_dim_; ++i) {
            const double lhs_y = static_cast<double>(codeValue(lhs_code, i)) - static_cast<double>(kUnsignedOffset);
            const double rhs_y = static_cast<double>(codeValue(rhs_code, i)) - static_cast<double>(kUnsignedOffset);
            code_ip += lhs_y * rhs_y;
        }

        const double residual_ip =
            0.25 * static_cast<double>(lhs_header.long_scale) *
            static_cast<double>(rhs_header.long_scale) * code_ip;
        return static_cast<float>(
            static_cast<double>(lhs_header.norm_sqr) + static_cast<double>(rhs_header.norm_sqr) -
            2.0 * residual_ip);
    }

 public:
    explicit RaBitQSpace(
        size_t dim,
        size_t centroid_count = 1,
        uint32_t random_seed = 100,
        bool external_residual_storage = false)
        : dim_(dim),
          code_dim_(roundUp64(dim)),
          compact_code_bytes_((code_dim_ * kTotalBits + 7U) / 8U),
          short_factor_bytes_(sizeof(ShortCodeFactors)),
          residual_factor_bytes_(sizeof(ResidualCodeFactors)),
          residual_block_count_((code_dim_ + kResidualBlockSize - 1U) / kResidualBlockSize),
          residual_scale_bytes_(residual_block_count_ * sizeof(float)),
          residual_code_bytes_(code_dim_),
          full_data_size_(sizeof(EncodedHeader) + short_factor_bytes_ + residual_factor_bytes_ +
              residual_scale_bytes_ + compact_code_bytes_ + residual_code_bytes_),
          residual_disk_record_bytes_(residual_factor_bytes_ + residual_scale_bytes_ + residual_code_bytes_),
          data_size_(external_residual_storage
              ? sizeof(EncodedHeader) + short_factor_bytes_ + compact_code_bytes_
              : full_data_size_),
          inv_sqrt_code_dim_(1.0f / std::sqrt(static_cast<float>(code_dim_))),
          external_residual_storage_(external_residual_storage),
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

    ~RaBitQSpace() override {
        closeExternalResidualStorage();
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

    size_t get_residual_code_bytes() const {
        return residual_code_bytes_;
    }

    size_t get_full_data_size() const {
        return full_data_size_;
    }

    size_t get_residual_disk_record_bytes() const {
        return residual_disk_record_bytes_;
    }

    bool external_residual_storage_enabled() const {
        return external_residual_storage_;
    }

    size_t get_centroid_count() const {
        return 1;
    }

    void saveState(std::ostream &output) const {
        const std::string magic = "EXRBTQ21";
        output.write(magic.data(), magic.size());

        const uint64_t dim = static_cast<uint64_t>(dim_);
        const uint64_t code_dim = static_cast<uint64_t>(code_dim_);
        const uint32_t total_bits = static_cast<uint32_t>(kTotalBits);
        const uint32_t remaining_bits = static_cast<uint32_t>(kRemainingBits);
        const uint64_t encoded_header_size = static_cast<uint64_t>(sizeof(EncodedHeader));
        const uint64_t short_factor_size = static_cast<uint64_t>(sizeof(ShortCodeFactors));
        const uint64_t residual_factor_size = static_cast<uint64_t>(sizeof(ResidualCodeFactors));
        const uint64_t residual_scale_bytes = static_cast<uint64_t>(residual_scale_bytes_);
        const uint64_t full_code_bytes = static_cast<uint64_t>(compact_code_bytes_);
        const uint64_t residual_code_bytes = static_cast<uint64_t>(residual_code_bytes_);
        const uint64_t data_size = static_cast<uint64_t>(data_size_);
        const uint64_t full_data_size = static_cast<uint64_t>(full_data_size_);
        const uint64_t residual_disk_record_bytes = static_cast<uint64_t>(residual_disk_record_bytes_);
        const uint8_t external_residual_storage = external_residual_storage_ ? 1U : 0U;
        output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        output.write(reinterpret_cast<const char *>(&code_dim), sizeof(code_dim));
        output.write(reinterpret_cast<const char *>(&total_bits), sizeof(total_bits));
        output.write(reinterpret_cast<const char *>(&remaining_bits), sizeof(remaining_bits));
        output.write(reinterpret_cast<const char *>(&encoded_header_size), sizeof(encoded_header_size));
        output.write(reinterpret_cast<const char *>(&short_factor_size), sizeof(short_factor_size));
        output.write(reinterpret_cast<const char *>(&residual_factor_size), sizeof(residual_factor_size));
        output.write(reinterpret_cast<const char *>(&residual_scale_bytes), sizeof(residual_scale_bytes));
        output.write(reinterpret_cast<const char *>(&full_code_bytes), sizeof(full_code_bytes));
        output.write(reinterpret_cast<const char *>(&residual_code_bytes), sizeof(residual_code_bytes));
        output.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));
        output.write(reinterpret_cast<const char *>(&full_data_size), sizeof(full_data_size));
        output.write(reinterpret_cast<const char *>(&residual_disk_record_bytes), sizeof(residual_disk_record_bytes));
        output.write(reinterpret_cast<const char *>(&external_residual_storage), sizeof(external_residual_storage));
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
        if (!input.good() || std::string(magic, sizeof(magic)) != "EXRBTQ21") {
            throw std::runtime_error(
                "Old or incompatible RaBitQ index format. Please rebuild the index.");
        }

        uint64_t stored_dim = 0;
        uint64_t stored_code_dim = 0;
        uint32_t stored_total_bits = 0;
        uint32_t stored_remaining_bits = 0;
        uint64_t stored_header_size = 0;
        uint64_t stored_factor_size = 0;
        uint64_t stored_residual_factor_size = 0;
        uint64_t stored_residual_scale_bytes = 0;
        uint64_t stored_full_code_bytes = 0;
        uint64_t stored_residual_code_bytes = 0;
        uint64_t stored_data_size = 0;
        uint64_t stored_full_data_size = 0;
        uint64_t stored_residual_disk_record_bytes = 0;
        uint8_t stored_external_residual_storage = 0;
        input.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        input.read(reinterpret_cast<char *>(&stored_code_dim), sizeof(stored_code_dim));
        input.read(reinterpret_cast<char *>(&stored_total_bits), sizeof(stored_total_bits));
        input.read(reinterpret_cast<char *>(&stored_remaining_bits), sizeof(stored_remaining_bits));
        input.read(reinterpret_cast<char *>(&stored_header_size), sizeof(stored_header_size));
        input.read(reinterpret_cast<char *>(&stored_factor_size), sizeof(stored_factor_size));
        input.read(reinterpret_cast<char *>(&stored_residual_factor_size), sizeof(stored_residual_factor_size));
        input.read(reinterpret_cast<char *>(&stored_residual_scale_bytes), sizeof(stored_residual_scale_bytes));
        input.read(reinterpret_cast<char *>(&stored_full_code_bytes), sizeof(stored_full_code_bytes));
        input.read(reinterpret_cast<char *>(&stored_residual_code_bytes), sizeof(stored_residual_code_bytes));
        input.read(reinterpret_cast<char *>(&stored_data_size), sizeof(stored_data_size));
        input.read(reinterpret_cast<char *>(&stored_full_data_size), sizeof(stored_full_data_size));
        input.read(reinterpret_cast<char *>(&stored_residual_disk_record_bytes), sizeof(stored_residual_disk_record_bytes));
        input.read(reinterpret_cast<char *>(&stored_external_residual_storage), sizeof(stored_external_residual_storage));

        if (!input.good()) {
            throw std::runtime_error("RaBitQSpace failed to read ExRaBitQ state header");
        }
        if (stored_dim != dim_ ||
            stored_code_dim != code_dim_ ||
            stored_total_bits != kTotalBits ||
            stored_remaining_bits != kRemainingBits ||
            stored_header_size != sizeof(EncodedHeader) ||
            stored_factor_size != sizeof(ShortCodeFactors) ||
            stored_residual_factor_size != sizeof(ResidualCodeFactors) ||
            stored_residual_scale_bytes != residual_scale_bytes_ ||
            stored_full_code_bytes != compact_code_bytes_ ||
            stored_residual_code_bytes != residual_code_bytes_ ||
            stored_data_size != data_size_ ||
            stored_full_data_size != full_data_size_ ||
            stored_residual_disk_record_bytes != residual_disk_record_bytes_ ||
            (stored_external_residual_storage != 0) != external_residual_storage_) {
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

    void closeExternalResidualStorage() const {
        if (residual_mmap_ != nullptr) {
            ::munmap(const_cast<char *>(residual_mmap_), residual_mmap_bytes_);
            residual_mmap_ = nullptr;
            residual_mmap_bytes_ = 0;
        }
        if (residual_fd_ >= 0) {
            ::close(residual_fd_);
            residual_fd_ = -1;
        }
    }

    void openExternalResidualStorage(const std::string &path, size_t record_count) const {
        if (!external_residual_storage_) {
            return;
        }
        closeExternalResidualStorage();
        residual_fd_ = ::open(path.c_str(), O_RDONLY);
        if (residual_fd_ < 0) {
            throw std::runtime_error("RaBitQ failed to open external residual file: " + path);
        }
        struct stat st;
        if (::fstat(residual_fd_, &st) != 0) {
            closeExternalResidualStorage();
            throw std::runtime_error("RaBitQ failed to stat external residual file: " + path);
        }
        const size_t expected_bytes = record_count * residual_disk_record_bytes_;
        if (static_cast<size_t>(st.st_size) != expected_bytes) {
            closeExternalResidualStorage();
            throw std::runtime_error("RaBitQ external residual file size mismatch: " + path);
        }
        if (expected_bytes == 0) {
            return;
        }
        void *mapped = ::mmap(nullptr, expected_bytes, PROT_READ, MAP_SHARED, residual_fd_, 0);
        if (mapped == MAP_FAILED) {
            closeExternalResidualStorage();
            throw std::runtime_error("RaBitQ failed to mmap external residual file: " + path);
        }
        residual_mmap_ = static_cast<const char *>(mapped);
        residual_mmap_bytes_ = expected_bytes;
#ifdef MADV_RANDOM
        ::madvise(const_cast<char *>(residual_mmap_), residual_mmap_bytes_, MADV_RANDOM);
#endif
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

        query->query_norm_sqr = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            residual[i] = raw_query[i] - global_center_[i];
            query->query_norm_sqr += residual[i] * residual[i];
        }

        query->rotated_residual.assign(code_dim_, 0.0f);
        rotate(residual.data(), query->rotated_residual);
        const size_t pair_count = code_dim_ >> 1U;
        query->rotated_residual_even.assign(pair_count, 0.0f);
        query->rotated_residual_odd.assign(pair_count, 0.0f);
        query->half_sum_residual = 0.0f;
        query->rotated_residual_sum = 0.0f;
        query->residual_nibble_lut_ready = false;
        for (size_t pair = 0; pair < pair_count; ++pair) {
            const float even_value = query->rotated_residual[pair << 1U];
            const float odd_value = query->rotated_residual[(pair << 1U) + 1U];
            query->rotated_residual_even[pair] = even_value;
            query->rotated_residual_odd[pair] = odd_value;
            query->half_sum_residual += even_value + odd_value;
            query->rotated_residual_sum += even_value + odd_value;
        }
        query->half_sum_residual *= 0.5f;

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
        const float short_ip = shortCodeIp(query, codeBytes(data_point));
        const float lower_bound = queryDistanceLowerBoundWithShortIp(query, header, factors, short_ip);
        if (lower_bound > threshold) {
            return false;
        }
        *distance = queryDistanceLongWithShortIp(query, data_point, header, short_ip);
        return true;
    }

    float result_distance(const void *prepared_query, const void *data_point) override {
        return query_distance(prepared_query, data_point);
    }

    bool supports_progressive_query_distance() const override {
        return true;
    }

    DistanceInterval compute_short_distance_interval(
        const void *prepared_query,
        const void *data_point) override {
        return computeShortDistanceIntervalTyped(
            *static_cast<const QueryContext *>(prepared_query),
            data_point);
    }

    DistanceInterval compute_long_distance_interval(
        const void *prepared_query,
        const void *data_point) override {
        return computeLongDistanceIntervalTyped(
            *static_cast<const QueryContext *>(prepared_query),
            data_point);
    }

    DistanceInterval compute_residual_distance_interval(
        const void *prepared_query,
        const void *data_point,
        float long_distance) override {
        return computeResidualDistanceIntervalTyped(
            *static_cast<const QueryContext *>(prepared_query),
            data_point,
            long_distance);
    }

    void batch_compute_residual_distance_intervals_by_id(
        const void *prepared_query,
        const size_t *internal_ids,
        const void *const *data_points,
        const float *long_distances,
        size_t count,
        DistanceInterval *intervals) override {
        const QueryContext &query = *static_cast<const QueryContext *>(prepared_query);
        for (size_t i = 0; i < count; ++i) {
            if (external_residual_storage_) {
                intervals[i] = computeResidualDistanceIntervalFromRecord(
                    query,
                    externalResidualRecord(internal_ids[i]),
                    long_distances[i]);
            } else {
                intervals[i] = computeResidualDistanceIntervalTyped(
                    query,
                    data_points[i],
                    long_distances[i]);
            }
        }
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
        thread_local std::vector<float> short_ips;
        short_ips.assign(count, 0.0f);
        batchShortCodeIp(query, data_points, count, short_ips.data());
        for (size_t i = 0; i < count; ++i) {
            const EncodedHeader header = loadHeader(data_points[i]);
            const ShortCodeFactors factors = *shortFactors(data_points[i]);
            lower_bounds[i] = queryDistanceLowerBoundWithShortIp(query, header, factors, short_ips[i]);
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
        thread_local std::vector<float> short_ips;
        short_ips.assign(count, 0.0f);
        batchShortCodeIp(query, data_points, count, short_ips.data());
        size_t survivor_count = 0;
        for (size_t i = 0; i < count; ++i) {
            const void *encoded = data_points[i];
            const EncodedHeader header = loadHeader(encoded);
            if (use_threshold) {
                const ShortCodeFactors factors = *shortFactors(encoded);
                const float lower_bound =
                    queryDistanceLowerBoundWithShortIp(query, header, factors, short_ips[i]);
                if (lower_bound > threshold) {
                    continue;
                }
            }
            distances[survivor_count] = queryDistanceLongWithShortIp(query, encoded, header, short_ips[i]);
            survivor_indices[survivor_count] = i;
            ++survivor_count;
        }
        return survivor_count;
    }

    void encodeVectorFull(const float *raw_vector, void *encoded_out) const {
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
        uint8_t *code = codeBytesFull(encoded_out);
        uint8_t *residual_code = residualCodeBytes(encoded_out);
        float *residual_scales = residualScales(encoded_out);
        ShortCodeFactors *factors = shortFactors(encoded_out);
        ResidualCodeFactors *residual_factors = residualFactors(encoded_out);
        header.norm_sqr = residual_norm * residual_norm;
        std::memset(code, 0, compact_code_bytes_);
        std::memset(residual_code, 0, residual_code_bytes_);
        std::memset(residual_scales, 0, residual_scale_bytes_);
        *factors = ShortCodeFactors{0.0f, 0.0f};
        *residual_factors = ResidualCodeFactors{0.0f, 0.0f};

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

        double o_obar = 0.0;
        double half_l1_norm = 0.0;
        const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(code_dim_));
        for (size_t i = 0; i < code_dim_; ++i) {
            const bool positive = rotated_unit[i] > 0.0f;
            const uint8_t magnitude = abs_code[i];
            const uint8_t packed_value = positive
                                             ? static_cast<uint8_t>(kMsbWeight + magnitude)
                                             : static_cast<uint8_t>(kRemainingMax - magnitude);
            setCodeValue(code, i, packed_value);

            const double sign = positive ? 1.0 : -1.0;
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
            const double fac_err_bound =
                std::ldexp(kExtendedRaBitQErrorConstant, -static_cast<int>(kTotalBits)) /
                std::sqrt(static_cast<double>(code_dim_));
            factors->error_scale = static_cast<float>(
                std::sqrt(std::max(0.0, (1.0 - o2) / o2)) * fac_err_bound * 2.0 * residual_norm);
        }

        thread_local std::vector<float> quantization_error;
        quantization_error.assign(code_dim_, 0.0f);
        for (size_t i = 0; i < code_dim_; ++i) {
            const double true_value = static_cast<double>(residual_norm) *
                                      static_cast<double>(rotated_unit[i]);
            const double x4_value = 0.5 * static_cast<double>(header.long_scale) *
                                    (static_cast<double>(codeValue(code, i)) -
                                     static_cast<double>(kUnsignedOffset));
            const double error = true_value - x4_value;
            quantization_error[i] = static_cast<float>(error);
        }
        double decoded_residual_norm_sqr = 0.0;
        double x4_residual_ip = 0.0;
        int8_t *residual_int8 = reinterpret_cast<int8_t *>(residual_code);
        for (size_t block = 0; block < residual_block_count_; ++block) {
            const size_t begin = block * kResidualBlockSize;
            const size_t end = std::min(code_dim_, begin + kResidualBlockSize);
            float max_abs_residual_error = 0.0f;
            for (size_t i = begin; i < end; ++i) {
                max_abs_residual_error =
                    std::max(max_abs_residual_error, static_cast<float>(std::abs(quantization_error[i])));
            }
            const float residual_scale = max_abs_residual_error > 0.0f
                ? max_abs_residual_error / 127.0f
                : 0.0f;
            residual_scales[block] = residual_scale;
            if (residual_scale == 0.0f || !std::isfinite(residual_scale)) {
                continue;
            }
            for (size_t i = begin; i < end; ++i) {
                const int quantized = static_cast<int>(std::round(
                    static_cast<double>(quantization_error[i]) /
                    static_cast<double>(residual_scale)));
                const int clipped = std::max(-127, std::min(127, quantized));
                residual_int8[i] = static_cast<int8_t>(clipped);
                const double decoded_residual =
                    static_cast<double>(residual_scale) * static_cast<double>(clipped);
                const double x4_value = 0.5 * static_cast<double>(header.long_scale) *
                                        (static_cast<double>(codeValue(code, i)) -
                                         static_cast<double>(kUnsignedOffset));
                decoded_residual_norm_sqr += decoded_residual * decoded_residual;
                x4_residual_ip += x4_value * decoded_residual;
            }
        }
        residual_factors->residual_norm_sqr = static_cast<float>(decoded_residual_norm_sqr);
        residual_factors->long_residual_inner_product = static_cast<float>(x4_residual_ip);
        std::memcpy(encoded_out, &header, sizeof(header));
    }

    void copyCompactPayloadFromFull(const void *full_encoded, void *compact_encoded) const {
        std::memcpy(compact_encoded, full_encoded, sizeof(EncodedHeader) + sizeof(ShortCodeFactors));
        std::memcpy(
            static_cast<char *>(compact_encoded) + sizeof(EncodedHeader) + sizeof(ShortCodeFactors),
            codeBytesFull(full_encoded),
            compact_code_bytes_);
    }

    void copyResidualRecordFromFull(const void *full_encoded, void *record_out) const {
        std::memcpy(record_out, residualFactors(full_encoded), residual_factor_bytes_);
        std::memcpy(static_cast<char *>(record_out) + residual_factor_bytes_, residualScales(full_encoded), residual_scale_bytes_);
        std::memcpy(
            static_cast<char *>(record_out) + residual_factor_bytes_ + residual_scale_bytes_,
            residualCodeBytes(full_encoded),
            residual_code_bytes_);
    }

    void encodeVector(const float *raw_vector, void *encoded_out) const {
        if (!external_residual_storage_) {
            encodeVectorFull(raw_vector, encoded_out);
            return;
        }
        thread_local std::vector<char> full_encoded;
        full_encoded.assign(full_data_size_, 0);
        encodeVectorFull(raw_vector, full_encoded.data());
        copyCompactPayloadFromFull(full_encoded.data(), encoded_out);
    }

    std::vector<char> encodeVector(const float *raw_vector) const {
        std::vector<char> encoded(data_size_, 0);
        encodeVector(raw_vector, encoded.data());
        return encoded;
    }
};

}  // namespace hnswlib
