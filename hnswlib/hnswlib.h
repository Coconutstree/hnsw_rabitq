#pragma once//防止头文件被多次包含

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)
//定义统一的错误输出口
#ifndef HNSWLIB_ERR_OVERRIDE
  #define HNSWERR std::cerr
#else
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

//判断编译阶段当前平台支不支持SSE/AVX指令集
#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK  0

static bool AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
    //labeltype是外部标签类型，tableint是内部id类型，linklistsizeint是连接列表大小类型
typedef size_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
//定义结果过滤器接口
class BaseFilterFunctor {
 public:
    virtual bool operator()(hnswlib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

template<typename dist_t>
//自定义停止接口，允许用户在搜索过程中根据距离和当前结果动态决定是否继续搜索
class BaseSearchStopCondition {
 public:
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_remove_extra() = 0;

    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

template <typename T>
//优先距离比较器
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

template<typename T>
//二元组距离比较器
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
//二元组距离比较器
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

template<typename MTYPE>
//定义了一个函数指针的类型别名，返回值类型是mystype，参数是三个void指针
//参数一是第一个向量的地址，参数二是第二个向量的地址，参数三是距离函数需要的额外参数（如维度信息等）的地址
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

template<typename MTYPE>
//距离空间接口
class SpaceInterface {
 public:
    // virtual void search(void *);
    //返回每个数据点占用多少字节
    virtual size_t get_data_size() = 0;
    //返回距离函数指针，也就是告诉算法怎么计算距离
    virtual DISTFUNC<MTYPE> get_dist_func() = 0;
    //返回距离函数参数的地址，告诉算法距离函数还需要哪些额外信息（如维度信息等）
    virtual void *get_dist_func_param() = 0;

    virtual const void *prepare_query(const void *query_data) {
        return query_data;
    }

    virtual void release_query(const void *prepared_query) {
        (void) prepared_query;
    }

    virtual MTYPE query_distance(const void *prepared_query, const void *data_point) {
        return get_dist_func()(prepared_query, data_point, get_dist_func_param());
    }

    virtual MTYPE result_distance(const void *prepared_query, const void *data_point) {
        return query_distance(prepared_query, data_point);
    }

    virtual bool supports_batch_query_distance() const {
        return false;
    }

    virtual void batch_query_distance(
        const void *prepared_query,
        const void *const *data_points,
        size_t count,
        MTYPE *distances) {
        for (size_t i = 0; i < count; ++i) {
            distances[i] = query_distance(prepared_query, data_points[i]);
        }
    }

    virtual ~SpaceInterface() {}
};

template<typename dist_t>
//统一算法接口，定义了添加数据点、搜索、保存索引等方法
class AlgorithmInterface {
 public:
 //添加数据点，参数是数据点的地址、标签和一个布尔值表示是否替换已删除的数据点
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;
//搜索k近邻，参数是查询点的地址、k值和一个可选的结果过滤器，返回一个优先队列，里面是距离和标签的二元组，距离较大的在前面
    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, labeltype>>
    //搜索k近邻，返回结果按照距离从近到远排序，参数同上
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;
//
    virtual void saveIndex(const std::string &location) = 0;
    virtual ~AlgorithmInterface(){
    }
};

template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
//searchKnnCLoserFirst具体实现的地方，它调用searchKnn获取结果，然后将结果从距离较大在前的优先队列转换成距离较小在前的vector
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        //获取结果的大小，调整result的大小，然后将结果从优先队列中弹出并放入result中，注意这里是从后往前放，所以result中的元素是按照距离从近到远排序的
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswlib
//欧氏距离空间实现
#include "space_l2.h"
//内积距离空间实现
#include "space_ip.h"
//停止条件具体实现
#include "stop_condition.h"
//暴力搜索实现
#include "bruteforce.h"
//hnsw图索引主体实现
#include "hnswalg.h"
