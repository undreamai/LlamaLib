// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <charconv>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP acc.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP acc.cuh
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP common.cuh
//
////////////////////////////////////////////////////////////////////////////////


#include "ggml.h"
#include "ggml-cuda.h"



#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#endif
#include "ggml-common.h"


#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#ifdef GGML_USE_TINYBLAS
#include "tinyblas.cu"
#define __nv_bfloat16 hip_bfloat16
#define CUBLAS_COMPUTE_16F TINYBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F TINYBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F TINYBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT TINYBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP TINYBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N TINYBLAS_OP_N
#define CUBLAS_OP_T TINYBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS TINYBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F TINYBLAS_R_16F
#define CUDA_R_32F TINYBLAS_R_32F
#define cublasComputeType_t tinyblasComputeType_t
#define cublasCreate tinyblasCreate
#define cublasDestroy tinyblasDestroy
#define cublasGemmEx tinyblasGemmEx
#define cublasGemmBatchedEx tinyblasGemmBatchedEx
#define cublasGemmStridedBatchedEx tinyblasGemmStridedBatchedEx
#define cublasHandle_t tinyblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream tinyblasSetStream
#define cublasSgemm tinyblasSgemm
#define cublasStatus_t tinyblasStatus_t
#define cudaDataType_t tinyblasDataType_t
#define cublasGetStatusString tinyblasGetStatusString

#else
#include <hipblas/hipblas.h>
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__
#define __nv_bfloat16 hip_bfloat16
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_32F  HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t //deprecated, new hipblasComputeType_t not in 5.6
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasStatus_t hipblasStatus_t
#define cudaDataType_t hipblasDatatype_t //deprecated, new hipblasDatatype not in 5.6
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#endif //GGML_USE_TINYBLAS

#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent(stream, event, flags) hipStreamWaitEvent(stream, event, flags)
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define __trap abort

#elif defined(GGML_USE_TINYBLAS)
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "tinyblas.cu"

#define CUBLAS_COMPUTE_16F TINYBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F TINYBLAS_COMPUTE_32F
#define CUBLAS_OP_N TINYBLAS_OP_N
#define CUBLAS_OP_T TINYBLAS_OP_T
#define CUDA_R_16F TINYBLAS_R_16F
#define CUDA_R_32F TINYBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT TINYBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP TINYBLAS_GEMM_DEFAULT
#define CUBLAS_STATUS_SUCCESS TINYBLAS_STATUS_SUCCESS
#define cublasGemmAlgo_t tinyblasGemmAlgo_t
#define cublasOperation_t tinyblasOperation_t
#define cublasComputeType_t tinyblasComputeType_t
#define cublasHandle_t tinyblasHandle_t
#define cublasStatus_t tinyblasStatus_t
#define cublasSgemm tinyblasSgemm
#define cublasGemmEx tinyblasGemmEx
#define cublasCreate tinyblasCreate
#define cublasDestroy tinyblasDestroy
#define cublasSetStream tinyblasSetStream
#define cublasGemmBatchedEx tinyblasGemmBatchedEx
#define cublasGemmStridedBatchedEx tinyblasGemmStridedBatchedEx
#define cublasGetStatusString tinyblasGetStatusString
#define cudaDataType_t tinyblasDataType_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS

#else
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

#endif // GGML_USE_HIP

#include "ggml-backend-impl.h"

#ifdef GGML_USE_TINYBLAS
#define BLAS_NAME "tinyBLAS"
#else
#define BLAS_NAME GGML_CUBLAS_NAME
#endif

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX   11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK  12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_PASCAL     600
#define GGML_CUDA_CC_DP4A       610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA      700
#define GGML_CUDA_CC_TURING     750
#define GGML_CUDA_CC_AMPERE     800
#define GGML_CUDA_CC_OFFSET_AMD 0x1000000

// GCN/CNDA, wave size is 64
#define GGML_CUDA_CC_GCN4       (GGML_CUDA_CC_OFFSET_AMD + 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA       (GGML_CUDA_CC_OFFSET_AMD + 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20     (GGML_CUDA_CC_OFFSET_AMD + 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_CDNA       (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_CUDA_CC_CDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x910)  // MI210, minimum acc register renameing
#define GGML_CUDA_CC_CDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300

// RNDA removes MFMA, dp4a, xnack, acc registers, wave size is 32
#define GGML_CUDA_CC_RDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x1010) // RX 5000
#define GGML_CUDA_CC_RDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x1030) // RX 6000, minimum for dp4a
#define GGML_CUDA_CC_RDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x1100) // RX 7000, minimum for WMMA

#define GGML_CUDA_CC_IS_RDNA(cc)  (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc) (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA2(cc) (cc >= GGML_CUDA_CC_RDNA2 && cc < GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_RDNA3(cc) (cc >= GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_GCN(cc)   (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_CDNA)
#define GGML_CUDA_CC_IS_CDNA(cc)  (cc >= GGML_CUDA_CC_CDNA && cc < GGML_CUDA_CC_RDNA1)

#define GGML_CUDA_CC_QY1        210
#define GGML_CUDA_CC_QY2        220

#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) {
    return false;
}

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur) {
    if (cur == 0) {
        GGML_ABORT("ggml was not compiled with any CUDA arch <= %d", arch);
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// ---------------------------------------------------------------------------------------------------------

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define GGML_CUDA_MAX_STREAMS 8

GGML_NORETURN
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

#if CUDART_VERSION >= 12000 || defined(GGML_USE_MUSA) || defined(GGML_USE_TINYBLAS)
    static const char * cublas_get_error_str(const cublasStatus_t err) {
#ifndef GGML_USE_MUSA
        return cublasGetStatusString(err);
#else
        return mublasStatus_to_string(err);
#endif // GGML_USE_MUSA
    }
#else
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }
#endif // CUDART_VERSION >= 12000

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#if !defined(GGML_USE_HIP)
static const char * cu_get_error_str(CUresult err) {
    const char * err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11010

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif //GGML_CUDA_F16

#if defined(GGML_USE_MUSA)
#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int __vsub4_musa(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4_musa(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4_musa(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}
#endif // defined(GGML_USE_MUSA)

#if defined(GGML_USE_HIPBLAS)
#define __CUDA_ARCH__ 1300

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#if defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}

#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
// __shfl_xor() for half2 was added in ROCm 5.6
static __device__ __forceinline__ half2 __shfl_xor(half2 var, int laneMask, int width) {
    typedef union half2_b32 {
        half2 val;
        int   b32;
    } half2_b32_t;
    half2_b32_t tmp;
    tmp.val = var;
    tmp.b32 = __shfl_xor(tmp.b32, laneMask, width);
    return tmp.val;
}
#endif // defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
#endif // defined(GGML_USE_HIPBLAS)

#if (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))
#define GGML_USE_VMM
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))

#if (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif // defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#define FP16_MMA_AVAILABLE
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define NEW_MMA_AVAILABLE
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ <= GGML_CUDA_CC_QY1)
#define FLASH_ATTN_AVAILABLE
#endif // !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ <= GGML_CUDA_CC_QY1)

static bool fp16_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_PASCAL;
}

static bool fast_fp16_available(const int cc) {
    return fp16_available(cc) && cc != 610;
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fast_fp16_hardware_available(const int cc) {
    return cc >= GGML_CUDA_CC_PASCAL && cc != 610;
}

// Any FP16 tensor core instructions are available for ggml code.
static bool fp16_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fp16_mma_hardware_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && cc >= GGML_CUDA_CC_VOLTA;
}

// Volta technically had FP16 tensor cores but they work very differently compared to Turing and later.
static bool new_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    return __AMDGCN_WAVEFRONT_SIZE;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
}

GGML_NORETURN
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#ifdef FP16_AVAILABLE
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    return __hmax(a, b);
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
#if defined(GGML_USE_HIP) && HIP_VERSION >= 50700000
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
#elif !defined(GGML_USE_HIP) && CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#elif !defined(GGML_USE_HIP)
    half2 ret;
    reinterpret_cast<half&>(ret.x) = __float2half(fmaxf( __low2float(a),  __low2float(b)));
    reinterpret_cast<half&>(ret.y) = __float2half(fmaxf(__high2float(a), __high2float(b)));
    return ret;
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || (defined(GGML_USE_HIP) && HIP_VERSION >= 50700000)
#pragma unroll
   for (int offset = width/2; offset > 0; offset >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
   }
   return x;
#else
   GGML_UNUSED(x);
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || (defined(GGML_USE_HIP) && HIP_VERSION >= 50700000)
}

#if CUDART_VERSION < CUDART_HMASK
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // CUDART_VERSION < CUDART_HMASK

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(RDNA2)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(__gfx1010__) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A

#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
}

// TODO: move to ggml-common.h
static constexpr __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, dfloat2 & v);

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
};

template<> // [jart] bf16
struct ggml_cuda_type_traits<GGML_TYPE_BF16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
};

//////////////////////

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        size_t  smpbo;              // max. shared memory per block (with opt-in)
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;          // Number of threads in a dispatch
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};


#if ((CUDART_VERSION >= 12000) && defined(GGML_CUDA_USE_GRAPHS)) || defined(GGML_HIP_GRAPHS)
#define USE_CUDA_GRAPH
#endif

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    std::vector<char **> updated_kernel_arg;
#endif
};

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

    std::unique_ptr<ggml_cuda_graph> cuda_graph;

    explicit ggml_backend_cuda_context(int device) :
        device(device),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_context() {
        if (copy_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(copy_event));
        }
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
                if (streams[i][j] != nullptr) {
                    CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
                }
            }
            if (cublas_handles[i] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
            }
        }
    }

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() {
        return stream(device, 0);
    }

    cublasHandle_t cublas_handle(int device) {
        if (cublas_handles[device] == nullptr) {
            ggml_cuda_set_device(device);
            CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
        }
        return cublas_handles[device];
    }

    cublasHandle_t cublas_handle() {
        return cublas_handle(device);
    }

    // pool
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device);

    ggml_cuda_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }
};

#define CUDA_ACC_BLOCK_SIZE 256

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
}

static void acc_f32_cuda(const float * x, const float * y, float * dst, const int n_elements,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, const int offset, cudaStream_t stream) {
    int num_blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    acc_f32<<<num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset);
}

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    acc_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], nb1, nb2, offset, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP arange.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP arange.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_ARANGE_BLOCK_SIZE 256

void ggml_cuda_op_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void arange_f32(float * dst, const int ne0, const float start, const float step) {
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    dst[nidx] = start + step * nidx;
}

static void arange_f32_cuda(float * dst, const int ne0, const float start, const float step, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_ARANGE_BLOCK_SIZE - 1) / CUDA_ARANGE_BLOCK_SIZE;
    arange_f32<<<num_blocks, CUDA_ARANGE_BLOCK_SIZE, 0, stream>>>(dst, ne0, start,  step);
}

void ggml_cuda_op_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float start;
    float stop;
    float step;
    memcpy(&start, (float *)dst->op_params + 0, sizeof(float));
    memcpy(&stop,  (float *)dst->op_params + 1, sizeof(float));
    memcpy(&step,  (float *)dst->op_params + 2, sizeof(float));

    int64_t steps = (int64_t)ceil((stop - start) / step);
    GGML_ASSERT(ggml_nelements(dst) == steps);

    arange_f32_cuda(dst_d, dst->ne[0], start, step, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP argsort.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP argsort.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP binbcast.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP binbcast.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13) {
    const int i0s = blockDim.x*blockIdx.x + threadIdx.x;
    const int i1 = (blockDim.y*blockIdx.y + threadIdx.y);
    const int i2 = (blockDim.z*blockIdx.z + threadIdx.z) / ne3;
    const int i3 = (blockDim.z*blockIdx.z + threadIdx.z) % ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x*gridDim.x) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13) {

    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}

template <typename T>
static __global__ void k_repeat_back(
    const T * __restrict__ src, T * __restrict__ dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {

    const int64_t tid0  = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t tid1  = int64_t(blockIdx.y)*blockDim.y + threadIdx.y;
    const int64_t tid23 = int64_t(blockIdx.z)*blockDim.z + threadIdx.z;
    const int64_t tid2  = tid23 % ne2;
    const int64_t tid3  = tid23 / ne2;

    if (tid0 >= ne0) {
        return;
    }

    T sum = 0;
    for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
        for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
            for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                    sum += src[i3*s03 + i2*s02 + i1*s01 + i0*s00];
                }
            }
        }
    }
    dst[tid3*ne2*ne1*ne0 + tid2*ne1*ne0 + tid1*ne0 + tid0] = sum;
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {

        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};

        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};

        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }

        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
            //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
            //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
            //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s00 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            dim3 block_dims;
            block_dims.x = std::min<unsigned int>(hne0, block_size);
            block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
            block_dims.z = std::min(std::min<unsigned int>(ne2*ne3, block_size / block_dims.x / block_dims.y), 64U);

            dim3 block_nums(
                (hne0 + block_dims.x - 1) / block_dims.x,
                (ne1 + block_dims.y - 1) / block_dims.y,
                (ne2*ne3 + block_dims.z - 1) / block_dims.z
            );

            if (block_nums.z > 65535) {
                // this is the maximum number of blocks in z dimension, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                k_bin_bcast_unravel<bin_op><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            } else {
                k_bin_bcast<bin_op><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            }
        }
    }
};

template <typename T>
static void repeat_back_cuda(
    const T * src, T * dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums((ne0 + WARP_SIZE - 1) / WARP_SIZE, ne1, ne2*ne3);
    k_repeat_back<T><<<block_nums, block_dims, 0, stream>>>
        (src, dst, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3);
}

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(dst, src0));

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(ne2*ne3 <= (1 << 15));

    const size_t ts = ggml_type_size(src0->type);
    const size_t s00 = nb00 / ts;
    const size_t s01 = nb01 / ts;
    const size_t s02 = nb02 / ts;
    const size_t s03 = nb03 / ts;

    switch (dst->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            float       * dst_d  = (float       *) dst->data;
            repeat_back_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        } break;
        default: {
            GGML_ASSERT(false);
        } break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP clamp.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP clamp.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_CLAMP_BLOCK_SIZE 256

void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void clamp_f32(const float * x, float * dst, const float min, const float max, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

static void clamp_f32_cuda(const float * x, float * dst, const float min, const float max, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    clamp_f32<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
}


void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    clamp_f32_cuda(src0_d, dst_d, min, max, ggml_nelements(src0), stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP concat.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP concat.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const float * src0_d = (const float *)src0->data;
        const float * src1_d = (const float *)src1->data;

        float * dst_d = (float *)dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d + i3 * (src0->nb[3] / 4),
                        src1_d + i3 * (src1->nb[3] / 4),
                        dst_d + i3 * ( dst->nb[3] / 4),
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,           src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/4, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_f32_non_cont<dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP convert.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP convert.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP dequantize.cuh
//
////////////////////////////////////////////////////////////////////////////////


static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(y);
    GGML_UNUSED(k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename src_t, typename dst_t>
static __global__ void convert_unary(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const src_t * x = (src_t *) vx;

    y[i] = x[i];
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP conv-transpose-1d.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP conv-transpose-1d.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static  __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        s0,p0,d0,output_size,
        src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
        src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
        src0,src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t kernel_size = ggml_nelements(src0);
    const int64_t input_size = ggml_nelements(src1);
    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_cuda(s0, p0, d0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0_d, src1_d, dst_d, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP cpy.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP cpy.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1);

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

static __device__ void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = __float2half(*xi);
}

static __device__ void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const half * xi = (const half *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const half * xi = (const half *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

// [jart] bf16
static __device__ void cpy_1_f32_bf16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    __nv_bfloat16 * dsti = (__nv_bfloat16 *) cdsti;
    *dsti = (__nv_bfloat16)(*xi);
}

// [jart] bf16
static __device__ void cpy_1_bf16_bf16(const char * cxi, char * cdsti) {
    const __nv_bfloat16 * xi = (const __nv_bfloat16 *) cxi;
    __nv_bfloat16 * dsti = (__nv_bfloat16 *) cdsti;
    *dsti = *xi;
}

// [jart] bf16
static __device__ void cpy_1_bf16_f32(const char * cxi, char * cdsti) {
    const __nv_bfloat16 * xi = (const __nv_bfloat16 *) cxi;
    float * dsti = (float *) cdsti;
    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static __global__ void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                   const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q8_0 * dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j]*id;

        dsti->qs[j] = roundf(x0);
    }
}

static __device__ void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    const block_q8_0 * xi = (const block_q8_0 *) cxi;
    float * dsti = (float *) cdsti;

    const float d = (float)xi->d;

    for (int j = 0; j < QK8_0; j++) {
       dsti[j] = xi->qs[j] * d;
    }
}

static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_0 * dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_1 * dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x = d;
    dsti->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (xi[0       + j] - vmin)*id;
        const float x1 = (xi[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q5_0 * dsti = (block_q5_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q5_1 * dsti = (block_q5_1 *) cdsti;

    float min = xi[0];
    float max = xi[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = xi[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x = d;
    dsti->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (xi[0       + j] - min)*id;
        const float x1 = (xi[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}


static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = xi[0        + j]*id;
        const float x1 = xi[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = xi[0        + j]*xi[0        + j];
        const float w1 = xi[QK4_NL/2 + j]*xi[QK4_NL/2 + j];
        sumqx += w0*v0*xi[j] + w1*v1*xi[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    dsti->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst, const int ne,
                                 const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                 const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                 const int nb12, const int nb13) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_q_f32(const char * cx, char * cdst, const int ne,
                                 const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                 const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                 const int nb12, const int nb13) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f16_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_bf16_f32_cuda( // [jart]
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_bf16_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_bf16_cuda( // [jart]
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_bf16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q8_0_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q8_0_f32, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f16_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f16_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_bf16_bf16_cuda( // [jart]
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_bf16_bf16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    if (src0->type == src1->type && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
        CUDA_CHECK(cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) { // [jart]
        ggml_cpy_bf16_bf16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) { // [jart]
        ggml_cpy_bf16_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) { // [jart]
        ggml_cpy_f32_bf16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1) {
    if (src0->type == src1->type && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        return nullptr;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_f32_f16<cpy_1_f32_f32>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_f32_f16<cpy_1_f32_f16>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>;
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        return (void*) cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        return (void*) cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        return (void*) cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_f32_f16<cpy_1_f32_f16>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_f32_f16<cpy_1_f16_f32>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        return (void*) cpy_f32_f16<cpy_1_f32_bf16>; // [jart]
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        return (void*) cpy_f32_f16<cpy_1_f32_bf16>; // [jart]
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_f32_f16<cpy_1_bf16_f32>; // [jart]
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP diagmask.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP diagmask.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.y*blockIdx.y + threadIdx.y;
    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void diag_mask_inf_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(nrows_x, block_num_x, 1);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int nrows0 = ggml_nrows(src0);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_cuda(src0_d, dst_d, ne00, nrows0, ne01, n_past, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mmv.cu
//
////////////////////////////////////////////////////////////////////////////////

#include "ggml.h"

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mmv.cuh
//
////////////////////////////////////////////////////////////////////////////////


// maximum number of src0 rows with which to use mul_mat_vec over cuBLAS if FP16 tensor cores are available
#define MMV_MAX_ROWS 512

void ggml_cuda_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

template <typename T, typename type_acc, int block_size>
static __global__ void mul_mat_vec(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst, const int64_t ncols2, const int64_t stride_row,
        const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst) {
    const int64_t row       = blockIdx.x;
    const int64_t channel   = blockIdx.y;
    const int64_t sample    = blockIdx.z;
    const int     tid       = threadIdx.x;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    x   +=  (sample/sample_ratio)*stride_sample_x   + (channel/channel_ratio)*stride_channel_x + row*stride_row;
    y   +=   sample              *stride_sample_y   +  channel               *stride_channel_y;
    dst +=   sample              *stride_sample_dst +  channel               *stride_channel_dst;

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf;

    if constexpr (std::is_same<T, half>::value) {
        const half2 * x2 = (const half2 *) x;

        if (std::is_same<type_acc, float>::value) {
            sumf = 0.0f;

            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                const float2 tmpy = y2[col2];
                sumf += tmpx.x * tmpy.x;
                sumf += tmpx.y * tmpy.y;
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2 = make_half2(0.0f, 0.0f);

            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmp = y2[col2];
                sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
            }

            sumf = __low2float(sumh2) + __high2float(sumh2);
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        const int * x2 = (const int *) x;
        sumf = 0.0f;

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const int    tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
        }
    } else {
        static_assert(std::is_same<T, void>::value, "unsupported type");
    }

    sumf = warp_reduce_sum<warp_size>(sumf);

    if (block_size > warp_size) {
        buf_iw[tid/warp_size] = sumf;
        __syncthreads();
        if (tid >= warp_size) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<warp_size>(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[row] = sumf;
}

template <typename T, typename type_acc>
static void launch_mul_mat_vec_cuda(
        const T * x, const float * y, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_y, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream) {
    GGML_ASSERT(ncols      % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);
    GGML_ASSERT(nchannels_y % nchannels_x == 0);
    GGML_ASSERT(nsamples_y  % nsamples_x  == 0);
    const int64_t channel_ratio = nchannels_y / nchannels_x;
    const int64_t sample_ratio  = nsamples_y  / nsamples_x;
    int device;
    int warp_size;

    CUDA_CHECK(cudaGetDevice(&device));
    warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const int smem = warp_size*sizeof(float);
    const dim3 block_nums(nrows, nchannels_y, nsamples_y);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec<T, type_acc,  32><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case   64: {
            mul_mat_vec<T, type_acc,  64><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case   96: {
            mul_mat_vec<T, type_acc,  96><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  128: {
            mul_mat_vec<T, type_acc, 128><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  160: {
            mul_mat_vec<T, type_acc, 160><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  192: {
            mul_mat_vec<T, type_acc, 192><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  224: {
            mul_mat_vec<T, type_acc, 224><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  256: {
            mul_mat_vec<T, type_acc, 256><<<block_nums, block_dims, smem, stream>>>
                (x, y, dst, ncols/2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template<typename T>
static void mul_mat_vec_cuda(
        const T * x, const float * y, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_y, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        enum ggml_prec prec, cudaStream_t stream) {
    switch (prec) {
        case GGML_PREC_DEFAULT: {
            launch_mul_mat_vec_cuda<T, half>
                (x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        } break;
        case GGML_PREC_F32: {
            launch_mul_mat_vec_cuda<T, float>
                (x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        } break;
    }
}

void ggml_cuda_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(ne12 == ne2);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0  == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    switch (src0->type) {
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_cuda(src0_d, src1_d, dst_d, ne00, ne01, s01, ne02, ne12, s02, s12, s2, ne03, ne13, s03, s13, s3, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_cuda(src0_d, src1_d, dst_d, ne00, ne01, s01, ne02, ne12, s02, s12, s2, ne03, ne13, s03, s13, s3, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(src1_ncols == 1);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;


    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_y         = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    switch (src0->type) {
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_cuda(src0_d, src1_ddf_i, dst_dd_i, ne00, row_diff, stride_row,
                nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_cuda(src0_d, src1_ddf_i, dst_dd_i, ne00, row_diff, stride_row,
                nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-common.cuh
//
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP vecdotq.cuh
//
////////////////////////////////////////////////////////////////////////////////


static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_16_q8_1_impl(
    const int * v, const int * u, const float * d8_0, const float & d8_1) {

    float sumf = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < vdr; i0 += QI8_0/2) {
        int sumi = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_0/2; ++i) {
            // SIMD dot product of quantized values
            sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
        }

        sumf += d8_0[i0/(QI8_0/2)]*sumi;
    }

    return d8_1*sumf;
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  4

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous v/x + u/y values
template <int ns8>
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8, const half2 * s8) {

    float sumf    = 0.0f;
    float sumf_d8 = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < QR2_K*VDR_Q2_K_Q8_1_MMQ; i0 += QI8_1) {
        const float2 dm2f0 = __half22float2(dm2[i0/(QI8_1/2) + 0]);
        int sumi_d0 = 0;

        const float2 dm2f1 = __half22float2(dm2[i0/(QI8_1/2) + 1]);
        int sumi_d1 = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d0 = ggml_cuda_dp4a(v[i], u[i], sumi_d0);
        }
        sumf_d8 += dm2f0.x * sumi_d0;

#pragma unroll
        for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
            sumi_d1 = ggml_cuda_dp4a(v[i], u[i], sumi_d1);
        }
        sumf_d8 += dm2f1.x * sumi_d1;

        if (i0/QI8_1 < ns8) {
            const float2 s8f = __half22float2(s8[i0/QI8_1]);
            sumf -= dm2f0.y*s8f.x;
            sumf -= dm2f1.y*s8f.y;
        } else {
            int sumi_m0 = 0;
#pragma unroll
            for (int i = i0; i < i0 + QI8_1/2; ++i) {
                sumi_m0 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m0);
            }
            sumf_d8 -= dm2f0.y * sumi_m0;

            int sumi_m1 = 0;
#pragma unroll
            for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
                sumi_m1 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m1);
            }
            sumf_d8 -= dm2f1.y * sumi_m1;
        }
    }

    return sumf + d8*sumf_d8;
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

    const int      sc_packed = get_int_b4(sc, 0);
    const int8_t * sc_reg    = (const int8_t *) &sc_packed;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc_reg[i0/2+0]*sumi_d.x + sc_reg[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b2(bq5_0->qs, iqs + i);
        vh[i]    = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq + kbx;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]    = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

#define VDR_IQ2_XXS_Q8_1_MMVQ 2
#define VDR_IQ2_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const int * grid_pos = (const int *) (iq2xxs_grid + aux8[k0/2]);
        const int signs_packed = ksigns_iq2xs[(aux32 >> (7*k0/2)) & 0x7F];

        const int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
        const int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
        const int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_XS_Q8_1_MMVQ 2
#define VDR_IQ2_XS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t * grid_pos = (const uint32_t *)(iq2xs_grid + (q2[l0/2] & 0x000001FF));
        const uint32_t * signs    = (const uint32_t *)(ksigns64   + (q2[l0/2] >> 9));

        const int grid_l = __vsub4(grid_pos[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid_pos[1] ^ signs[1], signs[1]);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_S_Q8_1_MMVQ 2
#define VDR_IQ2_S_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 2
#define VDR_IQ3_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);

        const int * signs = (const int *)(ksigns64 + ((aux32 >> (7*l0/2)) & 0x7F));

        const int grid_l = __vsub4(grid_pos.x ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid_pos.y ^ signs[1], signs[1]);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_S_Q8_1_MMVQ 2
#define VDR_IQ3_S_Q8_1_MMQ  2

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ1_S_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const float  d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
}

#define VDR_IQ1_M_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq1_m * bq1 = (const block_iq1_m *) vbq + kbx;

    const int       qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    int   sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4 * ((l0/2) % 2));

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi[l0/4] = ggml_cuda_dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, u1, sumi[l0/4]);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }

    const uint16_t * sc = (const uint16_t *) bq1->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);

    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4) {
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(
        kvalues_iq4nl[q0_8[0]], kvalues_iq4nl[q0_8[1]], kvalues_iq4nl[q0_8[2]], kvalues_iq4nl[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(
        kvalues_iq4nl[q1_8[0]], kvalues_iq4nl[q1_8[1]], kvalues_iq4nl[q1_8[2]], kvalues_iq4nl[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
}

#define VDR_IQ4_NL_Q8_1_MMVQ 2
#define VDR_IQ4_NL_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ 4
#define VDR_IQ4_XS_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3);

typedef half (*vec_dot_KQ_f16_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);
typedef float (*vec_dot_KQ_f32_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        const int v = (get_int_b2(K_q4_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 sum2 = __half2half2(K_q4_0[ib].d) * Q_ds[k_KQ_0/WARP_SIZE];
            sum += (T) (((half) sumi)*__low2half(sum2) - __high2half(sum2) /* *8/QI8_1 == 1 */);
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            sum += (T) (__half2float(K_q4_0[ib].d) * (sumi*Q_ds[k_KQ_0/WARP_SIZE].x - (8/QI8_1)*Q_ds[k_KQ_0/WARP_SIZE].y));
        }
    }

    return sum;
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        const int v = (get_int_b4(K_q4_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 d4d8_m4s8 = K_q4_1[ib].dm * Q_ds[k_KQ_0/WARP_SIZE];
            const half2 sumid4d8_m4s8scaled = d4d8_m4s8 * make_half2(sumi, 1.0f/QI8_1);
            sum += (T) (__low2half(sumid4d8_m4s8scaled) + __high2half(sumid4d8_m4s8scaled));
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            const float sumid4d8   =  __low2float(K_q4_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].x * sumi;
            const float m4s8scaled = __high2float(K_q4_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].y / QI8_1;

            sum += (T) (sumid4d8 + m4s8scaled);
        }
    }

    return sum;
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v = (get_int_b2(K_q5_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_b2(K_q5_0[ib].qh, 0) >> (iqs8 * QI5_0);
        v |= (vh <<  4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 sum2 = __half2half2(K_q5_0[ib].d) * Q_ds[k_KQ_0/WARP_SIZE];
            sum += (T) (((half) sumi)*__low2half(sum2) - __high2half(sum2)*__float2half(2.0f)) /* *16/QI8_1 == 2 */;
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            sum += (T) (__half2float(K_q5_0[ib].d) * (sumi*Q_ds[k_KQ_0/WARP_SIZE].x - (16/QI8_1)*Q_ds[k_KQ_0/WARP_SIZE].y));
        }
    }

    return sum;
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v = (get_int_b2(K_q5_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_b2(K_q5_1[ib].qh, 0) >> (iqs8 * QI5_1);
        v |= (vh <<  4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 d5d8_m5s8 = K_q5_1[ib].dm * Q_ds[k_KQ_0/WARP_SIZE];
            const half2 sumid5d8_m5s8scaled = d5d8_m5s8 * make_half2(sumi, 1.0f/QI8_1);
            sum += (T) (__low2half(sumid5d8_m5s8scaled) + __high2half(sumid5d8_m5s8scaled));
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            const float sumid5d8   =  __low2float(K_q5_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].x * sumi;
            const float m5s8scaled = __high2float(K_q5_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].y / QI8_1;

            sum += (T) (sumid5d8 + m5s8scaled);
        }
    }

    return sum;
}

template <typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        const int v = get_int_b2(K_q8_0[ib].qs, iqs);

        T Q_d;
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;
            Q_d = __low2half(Q_ds[k_KQ_0/WARP_SIZE]);
        } else {
            const float2 * Q_ds = (const float2 *) Q_ds_v;
            Q_d = Q_ds[k_KQ_0/WARP_SIZE].x;
        }

        sum += vec_dot_q8_0_q8_1_impl<T, 1>(&v, &Q_q8[k_KQ_0/WARP_SIZE], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        const half2 * Q_h2 = (const half2 *) Q_v;

        half2 sum2 = make_half2(0.0f, 0.0f);

#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
            const int k_KQ = k_KQ_0 + threadIdx.x;

            const half2 K_ik = K_h2[k_KQ];
            sum2 += K_ik * Q_h2[k_KQ_0/WARP_SIZE];
        }

        return __low2half(sum2) + __high2half(sum2);
    }
#endif // FP16_AVAILABLE

    const float2 * Q_f2 = (const float2 *) Q_v;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const half2 K_ik = K_h2[k_KQ];
        sum +=  __low2float(K_ik) * Q_f2[k_KQ_0/WARP_SIZE].x;
        sum += __high2float(K_ik) * Q_f2[k_KQ_0/WARP_SIZE].y;
    }

    return sum;
}

template <typename Tds>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < sizeof(int); ++l) {
        vals[l] = scale * x[4*threadIdx.x + l];
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < sizeof(int); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef half  (*dequantize_1_f16_t)(const void *, const int64_t);
typedef float (*dequantize_1_f32_t)(const void *, const int64_t);

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_0(const void * __restrict__ vx, const int64_t i) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i          /  QK4_0;
    const int     iqs   =  i          % (QK4_0/2);
    const int     shift = (i % QK4_0) / (QK4_0/2);

    const T   d  = x[ib].d;
    const int q0 = x[ib].qs[iqs];
    const int q  = ((q0 >> (4*shift)) & 0x0F) - 8;

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_1(const void * __restrict__ vx, const int64_t i) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i          /  QK4_1;
    const int     iqs   =  i          % (QK4_1/2);
    const int     shift = (i % QK4_1) / (QK4_1/2);

    const half2 dm = x[ib].dm;
    const int   q0 = x[ib].qs[iqs];
    const int   q  = ((q0 >> (4*shift)) & 0x0F);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm)*((half) q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm)*((float) q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_0(const void * __restrict__ vx, const int64_t i) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i          /  QK5_0;
    const int     idq   =  i          %  QK5_0;
    const int     iqs   =  i          % (QK5_0/2);
    const int     shift = (i % QK5_0) / (QK5_0/2);

    const T   d   = x[ib].d;
    const int ql0 = x[ib].qs[iqs];
    const int qh0 = get_int_b2(x[ib].qh, 0);
    const int ql  = ((ql0 >> (4*shift)) & 0x0F);
    const int qh  = ((qh0 >> idq) << 4) & 0x10;
    const int q   = (ql | qh) - 16;

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_1(const void * __restrict__ vx, const int64_t i) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i          /  QK5_1;
    const int     idq   =  i          %  QK5_1;
    const int     iqs   =  i          % (QK5_1/2);
    const int     shift = (i % QK5_1) / (QK5_1/2);

    const half2 dm  = x[ib].dm;
    const int   ql0 = x[ib].qs[iqs];
    const int   qh0 = get_int_b4(x[ib].qh, 0);
    const int   ql  = ((ql0 >> (4*shift)) & 0x0F);
    const int   qh  = ((qh0 >> idq) << 4) & 0x10;
    const int   q   = (ql | qh);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm)*((half) q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm)*((float) q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q8_0(const void * __restrict__ vx, const int64_t i) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i / QK8_0;
    const int     iqs = i % QK8_0;

    const T   d = x[ib].d;
    const int q = x[ib].qs[iqs];

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_f16(const void * __restrict__ vx, const int64_t i) {
    const half * x = (const half *) vx;

    return x[i];
}

template <int D>
constexpr __device__ vec_dot_KQ_f16_t get_vec_dot_KQ_f16(ggml_type type_K) {
    return type_K == GGML_TYPE_Q4_0 ? vec_dot_fattn_vec_KQ_q4_0<half, D> :
        type_K == GGML_TYPE_Q4_1 ? vec_dot_fattn_vec_KQ_q4_1<half, D> :
        type_K == GGML_TYPE_Q5_0 ? vec_dot_fattn_vec_KQ_q5_0<half, D> :
        type_K == GGML_TYPE_Q5_1 ? vec_dot_fattn_vec_KQ_q5_1<half, D> :
        type_K == GGML_TYPE_Q8_0 ? vec_dot_fattn_vec_KQ_q8_0<half, D> :
        type_K == GGML_TYPE_F16 ? vec_dot_fattn_vec_KQ_f16<half, D> :
        nullptr;
}

template <int D>
constexpr __device__ vec_dot_KQ_f32_t get_vec_dot_KQ_f32(ggml_type type_K) {
    return type_K == GGML_TYPE_Q4_0 ? vec_dot_fattn_vec_KQ_q4_0<float, D> :
        type_K == GGML_TYPE_Q4_1 ? vec_dot_fattn_vec_KQ_q4_1<float, D> :
        type_K == GGML_TYPE_Q5_0 ? vec_dot_fattn_vec_KQ_q5_0<float, D> :
        type_K == GGML_TYPE_Q5_1 ? vec_dot_fattn_vec_KQ_q5_1<float, D> :
        type_K == GGML_TYPE_Q8_0 ? vec_dot_fattn_vec_KQ_q8_0<float, D> :
        type_K == GGML_TYPE_F16 ? vec_dot_fattn_vec_KQ_f16<float, D> :
        nullptr;
}

constexpr __device__ dequantize_1_f16_t get_dequantize_1_f16(ggml_type type_V) {
    return type_V == GGML_TYPE_Q4_0 ? dequantize_1_q4_0<half> :
        type_V == GGML_TYPE_Q4_1 ? dequantize_1_q4_1<half> :
        type_V == GGML_TYPE_Q5_0 ? dequantize_1_q5_0<half> :
        type_V == GGML_TYPE_Q5_1 ? dequantize_1_q5_1<half> :
        type_V == GGML_TYPE_Q8_0 ? dequantize_1_q8_0<half> :
        type_V == GGML_TYPE_F16 ? dequantize_1_f16<half> :
        nullptr;
}

constexpr __device__ dequantize_1_f32_t get_dequantize_1_f32(ggml_type type_V) {
    return type_V == GGML_TYPE_Q4_0 ? dequantize_1_q4_0<float> :
        type_V == GGML_TYPE_Q4_1 ? dequantize_1_q4_1<float> :
        type_V == GGML_TYPE_Q5_0 ? dequantize_1_q5_0<float> :
        type_V == GGML_TYPE_Q5_1 ? dequantize_1_q5_1<float> :
        type_V == GGML_TYPE_Q8_0 ? dequantize_1_q8_0<float> :
        type_V == GGML_TYPE_F16 ? dequantize_1_f16<float> :
        nullptr;
}

// The HIP compiler for some reason complains that it can't unroll a loop because of the jt*ncols + j >= ne01 conditional.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template<int D, int ncols, int KQ_stride> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_stream_k_fixup(
        float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01, const int ne02, const int ne11) {
    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int iter_k = ne11 / KQ_stride;
    const int iter_j = (ne01 + (ncols - 1)) / ncols;

    const int bidx0 = blockIdx.x;

    const int kbc0      = (bidx0 + 0)*iter_k*iter_j*ne02 / gridDim.x;
    const int kbc0_stop = (bidx0 + 1)*iter_k*iter_j*ne02 / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    const int channel = kbc0 / (iter_k*iter_j);
    const int jt      = (kbc0 - channel*iter_k*iter_j) / iter_k;

    dst += jt*ncols*ne02*D + channel*D;

    // Load the partial result that needs a fixup:
    float dst_val[ncols] = {0.0f};
    float max_val[ncols] = {0.0f};
    float rowsum[ncols]  = {0.0f};
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (jt*ncols + j >= ne01) {
            break;
        }
        dst_val[j] = dst[j*ne02*D + threadIdx.x];

        const float2 tmp = dst_fixup[bidx0*ncols + j];
        max_val[j] = tmp.x;
        rowsum[j]  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = bidx*iter_k*iter_j*ne02 / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (jt*ncols + j >= ne01) {
                break;
            }
            const float dst_add = dst_fixup_data[bidx*ncols*D + j*D + threadIdx.x];

            const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + j];

            // Scale the current and new value accumulators depending on the max. values.
            const float max_val_new = fmaxf(max_val[j], tmp.x);

            const float diff_val = max_val[j] - max_val_new;
            const float diff_add = tmp.x      - max_val_new;

            const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
            const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

            dst_val[j] = scale_val*dst_val[j] + scale_add*dst_add;
            rowsum[j]  = scale_val*rowsum[j]  + scale_add*tmp.y;

            max_val[j] = max_val_new;
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (jt*ncols + j >= ne01) {
            return;
        }
        dst[j*ne02*D + threadIdx.x] = dst_val[j] / rowsum[j];
    }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template<int D, int parallel_blocks> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst) {
    VKQ_parts += parallel_blocks*D * gridDim.y*blockIdx.x;
    VKQ_meta  += parallel_blocks   * gridDim.y*blockIdx.x;
    dst       +=                 D * gridDim.y*blockIdx.x;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float2 meta[parallel_blocks];
    if (tid < 2*parallel_blocks) {
        ((float *) meta)[threadIdx.x] = ((const float *)VKQ_meta) [blockIdx.y*(2*parallel_blocks) + tid];
    }

    __syncthreads();

    float kqmax = meta[0].x;
#pragma unroll
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
#pragma unroll
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        const float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t *) &KQ_max_scale) &= ftz_mask;

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.y*D + blockIdx.y*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[blockIdx.y*D + tid] = VKQ_numerator / VKQ_denominator;
}

static void on_no_fattn_vec_case(const int D) {
    if (D == 64) {
        fprintf(stderr, "Unsupported KV type combination for head_size 64.\n");
        fprintf(stderr, "By default only f16 KV cache is supported.\n");
        fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for V cache quantization support.\n");
        GGML_ABORT("fatal error");
    } else if (D == 128) {
        fprintf(stderr, "Unsupported KV type combination for head_size 128.\n");
        fprintf(stderr, "Supported combinations:\n");
        fprintf(stderr, "  - K == q4_0, V == q4_0,  4.50 BPV\n");
        fprintf(stderr, "  - K == q8_0, V == q8_0,  8.50 BPV\n");
        fprintf(stderr, "  - K == f16,  V == f16,  16.00 BPV\n");
        fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.\n");
        GGML_ABORT("fatal error");
    } else {
        fprintf(stderr, "Unsupported KV type combination for head_size 256.\n");
        fprintf(stderr, "Only f16 is supported.\n");
        GGML_ABORT("fatal error");
    }
}

// parallel_blocks == 0 is stream-k decomposition
template <int D, int cols_per_block, int parallel_blocks, int KQ_stride>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel,
    const int nwarps, const size_t nbytes_shared, const bool need_f16_K, const bool need_f16_V
) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    GGML_ASSERT(Q->ne[3] == 1);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        K_f16.alloc(ggml_nelements(K));
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
        to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);
        K_data = (char *) K_f16.ptr;

        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        nb11 = nb11*bs*sizeof(half)/ts;
        nb12 = nb12*bs*sizeof(half)/ts;
        nb13 = nb13*bs*sizeof(half)/ts;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        V_f16.alloc(ggml_nelements(V));
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
        to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
        V_data = (char *) V_f16.ptr;

        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);

        nb21 = nb21*bs*sizeof(half)/ts;
        nb22 = nb22*bs*sizeof(half)/ts;
        nb23 = nb23*bs*sizeof(half)/ts;
    }

    const int ntiles_x = ((Q->ne[1] + cols_per_block - 1) / cols_per_block);
    const int ntiles_total = ntiles_x*Q->ne[2]*Q->ne[3];

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    dim3 blocks_num;
    if (parallel_blocks == 0) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int tiles_nwaves  = (ntiles_total - nsm - 1) / nsm;
        const bool tiles_inefficient = 3*nsm < 2*tiles_nwaves*ntiles_total;
        const bool short_context = K->ne[1] < 4096;

        const int nblocks_stream_k = 2*nsm;

        blocks_num.x = short_context && !tiles_inefficient ? ntiles_total : nblocks_stream_k;
        blocks_num.y = 1;
        blocks_num.z = 1;

        dst_tmp_meta.alloc(blocks_num.x*cols_per_block * (2*2 + D) * sizeof(float));
    } else {
        blocks_num.x = parallel_blocks*ntiles_x;
        blocks_num.y = Q->ne[2];
        blocks_num.z = Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }


    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        (parallel_blocks) > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
        mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
        Q->nb[1], Q->nb[2], Q->nb[3],
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
    );
    CUDA_CHECK(cudaGetLastError());

    if constexpr (parallel_blocks == 0) {
        if (blocks_num.x % ntiles_total != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(D, 1, 1);
            const dim3 blocks_num_combine = blocks_num;

            flash_attn_stream_k_fixup<D, cols_per_block, KQ_stride>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], K->ne[1]);
        }
    } else if constexpr (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], blocks_num.y, blocks_num.z);

        flash_attn_combine_results<D, parallel_blocks>
            <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
    }
    CUDA_CHECK(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-mma-f16.cuh
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mma.cuh
//
////////////////////////////////////////////////////////////////////////////////

// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape I x K.
// B is a column-major matrix with shape K x J.
// C is a column-major matrix with shape I x J.
// Note that along their lowest dimension I, J, and K are measured in physical 32 bit elements instead of logical elements.
// The functions get_i, get_j, and get_k can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.



#if CUDART_VERSION >= 11080

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    int ret = 0;

#ifdef NEW_MMA_AVAILABLE
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "+r"(ret) : "r"(x));
#else
    NO_DEVICE_CODE;
#endif // defined(NEW_MMA_AVAILABLE)
    return ret;
}

#else

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // CUDART_VERSION >= 11080


template <typename T>
struct mma_A_I16K4 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int I  = 16;
    static constexpr int K  = 4;
    static constexpr int ne = 2;

    T x[ne];

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l%2) * (I/2) + threadIdx.x / K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int *) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%I)*stride;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(xi[0]), "+r"(xi[1])
            : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_A_I16K8 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int I  = 16;
    static constexpr int K  = 8;
    static constexpr int ne = 4;

    T x[ne];

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l%2) * (I/2) + threadIdx.x / (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = (l/2) * (K/2) + threadIdx.x % (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int * ) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%I)*stride + (threadIdx.x/I)*(K/2);
        asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED(xs0);
        GGML_UNUSED(stride);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void load_ldmatrix_trans(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int * ) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%I)*stride + (threadIdx.x/I)*(K/2);
        asm("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(xi[0]), "+r"(xi[2]), "+r"(xi[1]), "+r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED(xs0);
        GGML_UNUSED(stride);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void transpose() {
        int * xi  = (int *) x;
        xi[0] = ggml_cuda_movmatrix(xi[0]);

        const int tmp = ggml_cuda_movmatrix(xi[1]);
        xi[1] = ggml_cuda_movmatrix(xi[2]);
        xi[2] = tmp;

        xi[3] = ggml_cuda_movmatrix(xi[3]);
    }
};

template <typename T>
struct mma_B_J8K4 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int J  = 8;
    static constexpr int K  = 4;
    static constexpr int ne = 1;

    T x[ne];

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int *) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%J)*stride;
        asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];"
            : "+r"(xi[0]) : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_B_J8K8 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int J  = 8;
    static constexpr int K  = 8;
    static constexpr int ne = 2;

    T x[ne];

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = l * (K/2) + threadIdx.x % (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int *) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%J)*stride + ((threadIdx.x/J)*(K/2)) % K;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(xi[0]), "+r"(xi[1])
            : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_C_I16J8 {};

template <>
struct mma_C_I16J8<int> {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = 4;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l/2) * (I/2) + threadIdx.x / (J/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J/2)) + l%2;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K4<int> & mma_A, const mma_B_J8K4<int> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<int> & mma_A, const mma_B_J8K8<int> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_A.x[2]), "r"(mma_A.x[3]), "r"(mma_B.x[0]), "r"(mma_B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[2]), "r"(mma_B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[3]), "r"(mma_B.x[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }
};

template <>
struct mma_C_I16J8<half2> {
    static constexpr int I  = 16;
    static constexpr int J  = 4;
    static constexpr int ne = 2;

    half2 x[ne] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = l * (I/2) + threadIdx.x / J;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x % J;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<half2> & mma_A, const mma_B_J8K8<half2> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
        int * Axi = (int *) mma_A.x;
        int * Bxi = (int *) mma_B.x;
        int * xi  = (int *) x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ mma_B_J8K8<half2> to_mma_B() {
        mma_B_J8K8<half2> mma_B;

        int * xi   = (int *) x;
        int * Bxi  = (int *) mma_B.x;
        Bxi[0] = ggml_cuda_movmatrix(xi[0]);
        Bxi[1] = ggml_cuda_movmatrix(xi[1]);

        return mma_B;
    }
};

template <>
struct mma_C_I16J8<float> {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = 4;

    float x[ne] = {0.0f, 0.0f, 0.0f, 0.0f};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l/2) * (I/2) + threadIdx.x / (J/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J/2)) + l%2;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<half2> & mma_A, const mma_B_J8K8<half2> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
        int * Axi = (int *) mma_A.x;
        int * Bxi = (int *) mma_B.x;
        int * xi  = (int *) x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ mma_B_J8K8<half2> to_mma_B() {
        mma_B_J8K8<half2> mma_B;
        mma_B.x[0] = make_half2(x[0], x[1]);
        mma_B.x[1] = make_half2(x[2], x[3]);

        int * Bxi  = (int *) mma_B.x;
        Bxi[0] = ggml_cuda_movmatrix(Bxi[0]);
        Bxi[1] = ggml_cuda_movmatrix(Bxi[1]);

        return mma_B;
    }

    __device__ __forceinline__ void load_generic(const float * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_i(l)];
        }
    }
};

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ maskh,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
#ifdef NEW_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    typedef mma_A_I16K8<half2> mma_A;
    typedef mma_B_J8K8<half2>  mma_B;
    typedef mma_C_I16J8<float> mma_C_KQ;
    typedef mma_C_I16J8<half2> mma_C_VKQ;

    static_assert(nwarps*mma_B::J % ncols == 0, "bad nwarps");
    constexpr int np = nwarps*mma_B::J / ncols; // Number of parallel CUDA warps per Q column.

    static_assert(D         % nwarps == 0, "bad D");
    static_assert(KQ_stride % nwarps == 0, "bad KQ_stride");

    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.
    extern __shared__ half2 tile_KV[]; // Temporary shared buffer for loading K/V data with KQ_stride*D logical elements.

    const int stride_Q    = nb01 / sizeof(float2);
    const int stride_KV   = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    mma_B Q_B[D/(2*mma_B::K)];
    mma_C_VKQ VKQ_C[D/mma_C_VKQ::I];

    float2    KQ_rowsum = {0.0f, 0.0f};
    float2       KQ_max = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
    float2 KQ_max_scale = {0.0f, 0.0f};

    // Temporarily load Q data into tile_KV, will be loaded into registers afterwards.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
        const int stride_j = WARP_SIZE / stride_k;

        if (nwarps*stride_j > ncols && threadIdx.y*stride_j >= ncols) {
            break;
        }

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps*stride_j) {
            const int j = j0 + threadIdx.y*stride_j + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jt*ncols + j < ne01) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols + j)*stride_Q + k];
                    tile_KV[j*D2_padded + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[j*D2_padded + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * mma_B::J;

#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += mma_B::K) {
            Q_B[k0/mma_B::K].load_ldmatrix(tile_KV + j0*D2_padded + k0, D2_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const int k_VKQ_0 = kb0*KQ_stride;
        mma_C_KQ KQ_C[KQ_stride/(np*mma_C_KQ::I)];

        // Load K data into tile with decreasing granularity for D for better memory bandwidth:
        static_assert(KQ_stride % (4*nwarps) == 0, "out of bounds");
#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
            const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < KQ_stride; i_KQ_0 += nwarps*stride_i) {
                const int i_KQ = i_KQ_0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += stride_k) {
                    const int k_KQ = k_KQ_0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[i_KQ*D2_padded + k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV + k_KQ];
                }
            }
        }

        __syncthreads();

        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < KQ_stride; i_KQ_00 += np*mma_A::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*mma_A::I;
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += mma_A::K) {
                mma_A K_A;
                K_A.load_ldmatrix(tile_KV + i_KQ_0*D2_padded + k_KQ_0, D2_padded);
                KQ_C[i_KQ_00/(np*mma_A::I)].mma(K_A, Q_B[k_KQ_0/mma_A::K]);
            }
        }

        __syncthreads();

        if (use_logit_softcap) {
            static_assert(KQ_stride % (np*mma_C_KQ::I) == 0, "bad loop size");
#pragma unroll
            for (int i = 0; i < KQ_stride/(np*mma_C_KQ::I); ++i) {
#pragma unroll
                for (int l = 0; l < mma_C_KQ::ne; ++l) {
                    KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
                }
            }
        }

        if (maskh) {
            static_assert(KQ_stride % (np       *mma_C_KQ::I) == 0, "bad loop size");
            static_assert(ncols     % (nwarps/np*mma_C_KQ::J) == 0, "bad loop size");
#pragma unroll
            for (int i00 = 0; i00 < KQ_stride; i00 += np*mma_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*mma_C_KQ::I;
#pragma unroll
                for (int l = 0; l < mma_C_KQ::ne; ++l) {
                    const int i = i0 + mma_C_KQ::get_i(l);
                    const int j = (threadIdx.y / np)*mma_C_KQ::J + mma_C_KQ::get_j(l);

                    KQ_C[i00/(np*mma_C_KQ::I)].x[l] += slope*__half2float(maskh[j*stride_mask + k_VKQ_0 + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        float2 KQ_max_new = KQ_max;
        static_assert(KQ_stride % (np*mma_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_stride/(np*mma_C_KQ::I); ++k) {
#pragma unroll
            for (int l0 = 0; l0 < mma_C_KQ::ne; l0 += 2) {
                KQ_max_new.x = fmaxf(KQ_max_new.x, KQ_C[k].x[l0 + 0]);
                KQ_max_new.y = fmaxf(KQ_max_new.y, KQ_C[k].x[l0 + 1]);
            }
        }

        // Values per KQ column are spread across 8 threads, does not need full warp reduce:
#pragma unroll
        for (int offset = 16; offset > 2; offset >>= 1) {
            KQ_max_new.x = fmaxf(KQ_max_new.x, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new.x, offset, WARP_SIZE));
            KQ_max_new.y = fmaxf(KQ_max_new.y, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new.y, offset, WARP_SIZE));
        }

        {
            const float2 diff = make_float2(KQ_max.x - KQ_max_new.x, KQ_max.y - KQ_max_new.y);
            KQ_max_scale = make_float2(expf(diff.x), expf(diff.y));
            if (diff.x <= SOFTMAX_FTZ_THRESHOLD) {
                KQ_max_scale.x = 0.0f;
            }
            if (diff.y <= SOFTMAX_FTZ_THRESHOLD) {
                KQ_max_scale.y = 0.0f;
            }
            KQ_max = KQ_max_new;
        }

        float2 KQ_rowsum_add = make_float2(0.0f, 0.0f);
        static_assert(KQ_stride % (np*mma_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_stride/(np*mma_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < mma_C_KQ::ne; ++l) {
                const float KQ_max_l = l % 2 == 0 ? KQ_max.x : KQ_max.y;
                const float diff = KQ_C[k].x[l] - KQ_max_l;
                KQ_C[k].x[l] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_C[k].x[l] = 0.0f;
                }

                if (l % 2 == 0) {
                    KQ_rowsum_add.x += KQ_C[k].x[l];
                } else {
                    KQ_rowsum_add.y += KQ_C[k].x[l];
                }
            }
        }

        // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
        KQ_rowsum.x = KQ_max_scale.x*KQ_rowsum.x + KQ_rowsum_add.x;
        KQ_rowsum.y = KQ_max_scale.y*KQ_rowsum.y + KQ_rowsum_add.y;

        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale.x, KQ_max_scale.y);
#pragma unroll
        for (int i = 0; i < D/mma_C_VKQ::I; ++i) {
#pragma unroll
            for (int l = 0; l < mma_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }

        // Convert KQ C tiles into B tiles for VKQ calculation:
        mma_B B[KQ_stride/(np*2*mma_B::K)];
        static_assert(KQ_stride % (np*2*mma_B::K) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_stride/(np*2*mma_B::K); ++k) {
            B[k] = KQ_C[k].to_mma_B();
        }

        // Load V data into tile with decreasing granularity for D for better memory bandwidth:
        static_assert(KQ_stride % (4*nwarps) == 0, "out of bounds");
#pragma unroll
        for (int stride_i : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int i0_start = stride_i == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_i);
            const int i0_stop  =                             D/2 - (D/2) % (1*stride_i);
            const int stride_k = WARP_SIZE / stride_i;

#pragma unroll
            for (int k_V_0 = 0; k_V_0 < KQ_stride; k_V_0 += nwarps*stride_k) {
                const int k_V = k_V_0 + threadIdx.y*stride_k + (stride_i == WARP_SIZE ? 0 : threadIdx.x / stride_i);

#pragma unroll
                for (int i_V_0 = i0_start; i_V_0 < i0_stop; i_V_0 += stride_i) {
                    const int i_V = i_V_0 + (stride_i == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_i);

                    tile_KV[k_V*D2_padded + i_V] = V_h2[(k_VKQ_0 + k_V)*stride_KV + i_V];
                }
            }
        }

        __syncthreads();

        // Calculate VKQ tile:
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += mma_C_VKQ::I) {
            static_assert((KQ_stride/2) % (np*mma_A::K) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < KQ_stride/2; k00 += np*mma_A::K) {
                const int k0 = k00 + (threadIdx.y % np)*mma_A::K;

                mma_A A;
                A.load_ldmatrix_trans(tile_KV + 2*k0*D2_padded + i_VKQ_0/2, D2_padded);
                VKQ_C[i_VKQ_0/mma_C_VKQ::I].mma(A, B[k00/(np*mma_A::K)]);
            }
        }

        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    // The partial sums are spread across 8 threads each, does not need full reduce.
#pragma unroll
    for (int offset = 16; offset > 2; offset >>= 1) {
        KQ_rowsum.x += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum.x, offset, WARP_SIZE);
        KQ_rowsum.y += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum.y, offset, WARP_SIZE);
    }

    // Write VKQ accumulators to shared memory in column-major format.
    // It's faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // Also for np > 1 the combination is done via these values in shared memory.
    const int j_cwd = threadIdx.y*mma_B::J + mma_B::get_j(-1); // j combine write data
#pragma unroll
    for (int k0 = 0; k0 < D/2; k0 += mma_B::K) {
        const mma_B B = VKQ_C[k0/mma_B::K].to_mma_B(); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int k = k0 + mma_B::get_k(l);

            tile_KV[j_cwd*D2_padded + k] = B.x[l];
        }
    }

    const int j_cwmo = (threadIdx.x % (2*mma_C_VKQ::J)) / mma_C_VKQ::J; // j combine write meta offset
    const int j_cwm = threadIdx.y*(2*mma_C_VKQ::J) + 2*mma_C_VKQ::get_j(-1) + j_cwmo; // j combine write meta
    const float2 KQ_cmr = make_float2(((const float *) &KQ_max)[j_cwmo], ((const float *) &KQ_rowsum)[j_cwmo]); // KQ combine max rowsum

    if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*mma_C_VKQ::J) {
        // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
        ((float2 *) tile_KV)[j_cwm*(D2_padded/2) + D/4] = KQ_cmr;
    }

    __syncthreads();

    static_assert(np == 1 || np == 2 || np == 4, "bad np");
    if (np == 1) {
        // No combination is needed, the meta data can be directly written from registers to VRAM.
        if (needs_fixup && threadIdx.x < mma_B::J) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[j_cwm] = KQ_cmr;
        }
        if (is_fixup && threadIdx.x < mma_B::J) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[j_cwm] = KQ_cmr;
        }
    } else if (threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        float * meta_j = (float *) tile_KV + (threadIdx.y*mma_B::J + threadIdx.x)*D2_padded + D/2;

        float KQ_cm = -FLT_MAX/2; // KQ combine max per parallel warp.
        if (np*mma_B::J == WARP_SIZE || threadIdx.x < np*mma_B::J) {
            KQ_cm = meta_j[0];
        }

        float KQ_cmn = KQ_cm; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int offset = np*mma_B::J/2; offset >= mma_B::J; offset >>= 1) {
            KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
        }

        const float KQ_cms = expf(KQ_cm - KQ_cmn); // KQ combine max scale per warp.
        float KQ_crs = 0.0f; // KQ combine rowsum, scaled sum of all parallel warps.
        if (np*mma_B::J == WARP_SIZE || threadIdx.x < np*mma_B::J) {
            KQ_crs = KQ_cms*meta_j[1];
        }
#pragma unroll
        for (int offset = np*mma_B::J/2; offset >= mma_B::J; offset >>= 1) {
            KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
        }

        // Write back combined meta data:
        if (np*mma_B::J == WARP_SIZE || threadIdx.x < np*mma_B::J) {
            meta_j[0] = KQ_cmn; // Combined max. KQ values.
            meta_j[1] = KQ_crs; // Combined KQ rowsums.
            meta_j[2] = KQ_cms; // KQ max scales per parallel warp.
        }
        if (needs_fixup && threadIdx.x < mma_B::J) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*mma_B::J + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && threadIdx.x < mma_B::J) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*mma_B::J + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    }

    if (np > 1) {
        __syncthreads();
    }

    if (np == 1 || threadIdx.y % np == 0) {
        // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
        // The values after that are for the partial results of the individual blocks.
        float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(D/2));

#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
            const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
            const int stride_j = WARP_SIZE / stride_k;

            if (nwarps*stride_j > ncols && threadIdx.y*stride_j >= ncols) {
                break;
            }

#pragma unroll
            for (int j0_dst = 0; j0_dst < ncols; j0_dst += (nwarps/np)*stride_j) {
                const int j_dst = j0_dst + (threadIdx.y/np)*stride_j + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);
                const int j_tile_KV = (j_dst/mma_B::J)*(np*mma_B::J) + j_dst % mma_B::J;

                if (!is_fixup && jt*ncols + j_dst >= ne01) {
                    continue;
                }
                const float * meta_j = (const float *) tile_KV + j_tile_KV*D2_padded + D/2;
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*mma_B::J*D2_padded + 2];
                        const float2 dstk_val_add = __half22float2(tile_KV[(j_tile_KV + ip*mma_B::J)*D2_padded + k]);
                        dstk_val.x += dstk_val_add.x*KQ_crs;
                        dstk_val.y += dstk_val_add.y*KQ_crs;
                    }

                    if (!needs_fixup && !is_fixup) {
                        const float KQ_rowsum_j = meta_j[1];
                        dstk_val.x /= KQ_rowsum_j;
                        dstk_val.y /= KQ_rowsum_j;
                    }

                    if (is_fixup) {
                        dstk_fixup_data[j_dst*(D/2) + k] = dstk_val;
                    } else {
                        dstk[(jt*ncols + j_dst)*ne02*(D/2) + k] = dstk_val;
                    }
                }
            }
        }
    }

    if (np > 1) {
        __syncthreads();
    }
#else
   NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

#ifndef GGML_MINIMIZE_CODE_SIZE

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap>
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 2)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(FATTN_KQ_STRIDE % KQ_stride == 0, "bad KQ_stride");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int iter_k = ne11 / KQ_stride;
    const int iter_j = (ne01 + (ncols - 1)) / ncols;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*iter_k*iter_j*ne02 / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*iter_k*iter_j*ne02 / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int channel = kbc / (iter_k*iter_j);
        const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

        const float2 * Q_f2  = (const float2 *) (Q + nb02* channel);
        const half2  * K_h2  = (const half2  *) (K + nb12*(channel / gqa_ratio));
        const half2  * V_h2  = (const half2  *) (V + nb12*(channel / gqa_ratio)); // K and V have same shape
        const half   * maskh = mask ? (const half  *) mask + (nb31/sizeof(half))*jt*ncols : nullptr;
        float2       * dstk  = ((float2 *) dst) + channel*(D/2);

        const float slope = get_alibi_slope(max_bias, channel, n_head_log2, m0, m1);

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne31, nb31, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne0, ne1, ne2, ne3,
                jt, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne31, nb31, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne0, ne1, ne2, ne3,
                jt, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int channel = kbc / (iter_k*iter_j);
    const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

    const float2 * Q_f2  = (const float2 *) (Q + nb02* channel);
    const half2  * K_h2  = (const half2  *) (K + nb12*(channel / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V + nb12*(channel / gqa_ratio)); // K and V have same shape
    const half   * maskh = mask ? (const half  *) mask + (nb31/sizeof(half))*jt*ncols : nullptr;
    float2       * dstk  = ((float2 *) dst) + channel*(D/2);

    const float slope = get_alibi_slope(max_bias, channel, n_head_log2, m0, m1);

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
        ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne31, nb31, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne0, ne1, ne2, ne3,
        jt, kb0_start, kb0_stop);
}

template <int D, int cols_per_block>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    typedef mma_A_I16K8<half2> mma_A;
    typedef mma_B_J8K8<half2>  mma_B;

    static_assert(D              % mma_B::K == 0, "bad D");
    static_assert(cols_per_block % mma_B::J == 0, "bad cols_per_block");

    const ggml_tensor * KQV = dst;

    constexpr int    KQ_stride     = D <= 128 ? 64 : 32;
    constexpr int    nwarps        = (KQ_stride == 32 && cols_per_block <= 16) ?
                                     cols_per_block/mma_B::J * KQ_stride/mma_A::I : (cols_per_block <= 8 ? 4 : 8);
    constexpr size_t nbytes_shared = std::max(KQ_stride, nwarps*mma_B::J) * (D + 8) * sizeof(half);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<D, cols_per_block, nwarps, KQ_stride, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<D, cols_per_block, nwarps, KQ_stride, use_logit_softcap>;
    }
    launch_fattn<D, cols_per_block, 0, KQ_stride>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
}

#define DECL_FATTN_MMA_F16_CASE(D, cols_per_block)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                     \
    <D, cols_per_block>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_MMA_F16_CASE( 64,  8);
extern DECL_FATTN_MMA_F16_CASE( 80,  8);
extern DECL_FATTN_MMA_F16_CASE( 96,  8);
extern DECL_FATTN_MMA_F16_CASE(112,  8);
extern DECL_FATTN_MMA_F16_CASE(128,  8);
extern DECL_FATTN_MMA_F16_CASE(256,  8);

extern DECL_FATTN_MMA_F16_CASE( 64, 16);
extern DECL_FATTN_MMA_F16_CASE( 80, 16);
extern DECL_FATTN_MMA_F16_CASE( 96, 16);
extern DECL_FATTN_MMA_F16_CASE(112, 16);
extern DECL_FATTN_MMA_F16_CASE(128, 16);
extern DECL_FATTN_MMA_F16_CASE(256, 16);

extern DECL_FATTN_MMA_F16_CASE( 64, 32);
extern DECL_FATTN_MMA_F16_CASE( 80, 32);
extern DECL_FATTN_MMA_F16_CASE( 96, 32);
extern DECL_FATTN_MMA_F16_CASE(112, 32);
extern DECL_FATTN_MMA_F16_CASE(128, 32);
extern DECL_FATTN_MMA_F16_CASE(256, 32);

extern DECL_FATTN_MMA_F16_CASE( 64, 64);
extern DECL_FATTN_MMA_F16_CASE( 80, 64);
extern DECL_FATTN_MMA_F16_CASE( 96, 64);
extern DECL_FATTN_MMA_F16_CASE(112, 64);
extern DECL_FATTN_MMA_F16_CASE(128, 64);
extern DECL_FATTN_MMA_F16_CASE(256, 64);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-tile-f16.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_flash_attn_ext_tile_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-tile-f32.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-vec-f16.cuh
//
////////////////////////////////////////////////////////////////////////////////


template<int D, int ncols, int parallel_blocks, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#ifdef FP16_AVAILABLE

#ifndef FLASH_ATTN_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f16_t vec_dot_KQ = get_vec_dot_KQ_f16<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f16_t dequantize_1_v = get_dequantize_1_f16(type_V);

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb02* blockIdx.y              + nb01*ic0;
    K += nb12*(blockIdx.y / gqa_ratio);
    V += nb22*(blockIdx.y / gqa_ratio);

    const half * maskh = (const half   *)  mask + ne11*ic0;

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half KQ[ncols*D];
    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
    }
    half kqsum[ncols] = {0.0f};

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][D/(2*WARP_SIZE)];
    int   Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D/(sizeof(int)*QK8_1)];
    half2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }


#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -HALF_MAX_HALF;
    }

    half2 VKQ[ncols] = {{0.0f, 0.0f}};

    const int k_start = parallel_blocks == 1 ? 0 : ip*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + (k_VKQ_0 + i_KQ)*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += mask ? slopeh*maskh[j*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k_VKQ_0 + k0 + 0)*nb21, tid);
            reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k_VKQ_0 + k0 + 1)*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum((float)kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum((float)kqsum[j_VKQ]);

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (parallel_blocks == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
        dst[j_dst*D*gridDim.y + D*blockIdx.y + tid] = dst_val;
    }

    if (parallel_blocks != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[(ic0 + tid)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int D, int cols_per_block, int parallel_blocks, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, need_f16_K, need_f16_V);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block  = 1;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block  = 2;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block  = 4;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 8) {
        constexpr int cols_per_block  = 8;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block  = 8;
    constexpr int parallel_blocks = 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F16_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f16_case                     \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-vec-f32.cuh
//
////////////////////////////////////////////////////////////////////////////////


template<int D, int ncols, int parallel_blocks, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#ifndef FLASH_ATTN_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f32_t vec_dot_KQ = get_vec_dot_KQ_f32<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f32_t dequantize_1_v = get_dequantize_1_f32(type_V);

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb02* blockIdx.y              + nb01*ic0;
    K += nb12*(blockIdx.y / gqa_ratio);
    V += nb22*(blockIdx.y / gqa_ratio); // K and V have same shape
    const half * maskh = (const half   *)  mask + ne11*ic0;

    const float slope = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float KQ[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -FLT_MAX/2.0f;
    }

    float kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols] = {0.0f};

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    float2  Q_f2[ncols][D/(2*WARP_SIZE)];
    int    Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D >= D/(sizeof(int)*QK8_1)];
    float2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_f2[j][i0/WARP_SIZE]    = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_f2[j][i0/WARP_SIZE].x *= scale;
                Q_f2[j][i0/WARP_SIZE].y *= scale;
            }
        }
    }

    float VKQ[ncols] = {0.0f};

    const int k_start = parallel_blocks == 1 ? 0 : ip*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + (k_VKQ_0 + i_KQ)*nb11, Q_f2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += mask ? slope*__half2float(maskh[j*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < D; ++k) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            const float V_ki = dequantize_1_v(V + (k_VKQ_0 + k)*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_ki*KQ[j*D + k];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        float dst_val = VKQ[j_VKQ];
        if (parallel_blocks == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
        dst[j_dst*D*gridDim.y + D*blockIdx.y + tid] = dst_val;
    }

    if (parallel_blocks != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[(ic0 + tid)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[tid], kqsum[tid]);
    }
}

template <int D, int cols_per_block, int parallel_blocks, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f32_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, need_f16_K, need_f16_V);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block  = 1;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block  = 2;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block  = 4;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 8) {
        constexpr int cols_per_block  = 8;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block  = 8;
    constexpr int parallel_blocks = 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, parallel_blocks, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F32_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f32_case                     \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-wmma-f16.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int cols_per_block>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_hs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    switch (Q->ne[0]) {
        case 64:
            ggml_cuda_flash_attn_ext_mma_f16_case< 64, cols_per_block>(ctx, dst);
            break;
        case 80:
            ggml_cuda_flash_attn_ext_mma_f16_case< 80, cols_per_block>(ctx, dst);
            break;
        case 96:
            ggml_cuda_flash_attn_ext_mma_f16_case< 96, cols_per_block>(ctx, dst);
            break;
        case 112:
            ggml_cuda_flash_attn_ext_mma_f16_case<112, cols_per_block>(ctx, dst);
            break;
        case 128:
            ggml_cuda_flash_attn_ext_mma_f16_case<128, cols_per_block>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_mma_f16_case<256, cols_per_block>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] <= 8) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<8>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 16) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<16>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 32) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<32>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_hs<64>(ctx, dst);
}

#define FATTN_VEC_F16_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16 )

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(Q->ne[0]);
}

#define FATTN_VEC_F32_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(Q->ne[0]);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= GGML_CUDA_CC_OFFSET_AMD) {
        if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (prec == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    if (Q->ne[1] == 1 && Q->ne[0] % (2*WARP_SIZE) == 0) {
        if (prec == GGML_PREC_DEFAULT) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            return;
        } else if(Q->ne[0] <= 128) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            return;
        }
    }

    // The MMA implementation needs Turing or newer, use the old WMMA code for Volta:
    if (cc == GGML_CUDA_CC_VOLTA) {
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-tile-f16.cu
//
////////////////////////////////////////////////////////////////////////////////


#define FATTN_KQ_STRIDE_TILE_F16 64

template<int D, int ncols, int nwarps, int parallel_blocks, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#ifdef FP16_AVAILABLE

#ifndef FLASH_ATTN_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ half KQ[ncols*FATTN_KQ_STRIDE_TILE_F16];
    half2 * KQ2 = (half2 *) KQ;

    __shared__ half2 KV_tmp[FATTN_KQ_STRIDE_TILE_F16][D/2 + 1]; // Pad D to avoid memory bank conflicts.

    half kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -HALF_MAX_HALF;
    }
    half2 kqsum[ncols/nwarps] = {{0.0f, 0.0f}};

    half2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ half2 Q_h2[ncols][D/2];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            Q_h2[j][i] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    __syncthreads();

    const int k_start = parallel_blocks == 1 ? 0 : ip*FATTN_KQ_STRIDE_TILE_F16;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE_TILE_F16) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        half kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                KV_tmp[i_KQ][k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
            }
        }

        __syncthreads();

        half2 sum2[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE][ncols/nwarps] = {{{0.0f, 0.0f}}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D/2; ++k_KQ) {
            half2 K_k[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE];
            half2 Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_h2[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE]*Q_k[j_KQ_0/nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                half sum;
                if (use_logit_softcap) {
                    const float2 tmp = __half22float2(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                    sum = logit_softcap * tanhf(tmp.x + tmp.y);
                } else {
                    sum = __low2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]) + __high2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }
                sum += mask ? slopeh*maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                kqmax_new[j_KQ_0/nwarps] = ggml_cuda_hmax(kqmax_new[j_KQ_0/nwarps], sum);

                KQ[j_KQ*FATTN_KQ_STRIDE_TILE_F16 + i_KQ] = sum;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const half2 KQ_max_scale = __half2half2(hexp(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]));
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F16/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const half2 diff = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + i] - __half2half2(kqmax[j0/nwarps]);
                const half2 val = h2exp(diff);
                kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + val;
                KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + i] = val;
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE] *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp[k][i] = V_h2[(k_VKQ_0 + k)*stride_KV2 + i];
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += 2) {
            half2  V_k[(D/2)/WARP_SIZE][2];
            half2 KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE][0] = KV_tmp[k0 + 0][i];
                V_k[i0/WARP_SIZE][1] = KV_tmp[k0 + 1][i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + k0/2];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][0]* __low2half2(KQ_k[j0/nwarps]);
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][1]*__high2half2(KQ_k[j0/nwarps]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        half kqsum_j = __low2half(kqsum[j_VKQ_0/nwarps]) + __high2half(kqsum[j_VKQ_0/nwarps]);
        kqsum_j = warp_reduce_sum((float)kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            half2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (parallel_blocks == 1) {
                dst_val /= __half2half2(kqsum_j);
            }
            const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 0] =  __low2float(dst_val);
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 1] = __high2float(dst_val);
        }

        if (parallel_blocks != 1 && threadIdx.x == 0) {
            dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int cols_per_block, int parallel_blocks, bool use_logit_softcap>
void launch_fattn_tile_f16_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int    D             = 64;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, parallel_blocks, use_logit_softcap>;
            launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
        } break;
        case 128: {
            constexpr int    D             = 128;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, parallel_blocks, use_logit_softcap>;
            launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 32;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int parallel_blocks = 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-tile-f32.cu
//
////////////////////////////////////////////////////////////////////////////////


#define FATTN_KQ_STRIDE_TILE_F32 32

template<int D, int ncols, int nwarps, int parallel_blocks, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#ifndef FLASH_ATTN_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slope = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ float KQ[ncols*FATTN_KQ_STRIDE_TILE_F32];

    __shared__ float KV_tmp[FATTN_KQ_STRIDE_TILE_F32][D + 1]; // Pad D to avoid memory bank conflicts.
    float2 * KV_tmp2 = (float2 *) KV_tmp;

    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols/nwarps] = {0.0f};

    float2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ float Q_f[ncols][D];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += 2*WARP_SIZE) {
            float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i0/2 + threadIdx.x] : make_float2(0.0f, 0.0f);
            Q_f[j][i0 + 0*WARP_SIZE + threadIdx.x] = tmp.x * scale;
            Q_f[j][i0 + 1*WARP_SIZE + threadIdx.x] = tmp.y * scale;
        }
    }

    __syncthreads();

    const int k_start = parallel_blocks == 1 ? 0 : ip*FATTN_KQ_STRIDE_TILE_F32;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE_TILE_F32) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 2*WARP_SIZE) {
                const half2 tmp = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ_0/2 + threadIdx.x];
                KV_tmp[i_KQ][k_KQ_0 + 0*WARP_SIZE + threadIdx.x] =  __low2float(tmp);
                KV_tmp[i_KQ][k_KQ_0 + 1*WARP_SIZE + threadIdx.x] = __high2float(tmp);
            }
        }

        __syncthreads();

        float sum[FATTN_KQ_STRIDE_TILE_F32/WARP_SIZE][ncols/nwarps] = {{0.0f}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D; ++k_KQ) {
            float K_k[FATTN_KQ_STRIDE_TILE_F32/WARP_SIZE];
            float Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_f[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE] * Q_k[j_KQ_0/nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                if (use_logit_softcap) {
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] = logit_softcap * tanhf(sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }

                sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += mask ? slope*__half2float(maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);

                KQ[j_KQ*FATTN_KQ_STRIDE_TILE_F32 + i_KQ] = sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

            float kqsum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F32; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float diff = KQ[j*FATTN_KQ_STRIDE_TILE_F32 + i] - kqmax[j0/nwarps];
                const float val = expf(diff);
                kqsum_add += val;
                KQ[j*FATTN_KQ_STRIDE_TILE_F32 + i] = val;
            }
            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + kqsum_add;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/WARP_SIZE].y *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F32; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp2[k*(D/2) + i].x =  __low2float(V_h2[(k_VKQ_0 + k)*stride_KV2 + i]);
                KV_tmp2[k*(D/2) + i].y = __high2float(V_h2[(k_VKQ_0 + k)*stride_KV2 + i]);
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < FATTN_KQ_STRIDE_TILE_F32; ++k) {
            float2 V_k[(D/2)/WARP_SIZE];
            float  KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE] = KV_tmp2[k*(D/2) + i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ[j*FATTN_KQ_STRIDE_TILE_F32 + k];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0/nwarps][i0/WARP_SIZE].x += V_k[i0/WARP_SIZE].x*KQ_k[j0/nwarps];
                    VKQ[j0/nwarps][i0/WARP_SIZE].y += V_k[i0/WARP_SIZE].y*KQ_k[j0/nwarps];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float kqsum_j = kqsum[j_VKQ_0/nwarps];
        kqsum_j = warp_reduce_sum(kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            float2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (parallel_blocks == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 0] = dst_val.x;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 1] = dst_val.y;
        }

        if (parallel_blocks != 1 && threadIdx.x == 0) {
            dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
}

template <int cols_per_block, int parallel_blocks, bool use_logit_softcap>
void launch_fattn_tile_f32_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int    D             = 64;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, parallel_blocks, use_logit_softcap>;
            launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
        } break;
        case 128: {
            constexpr int    D             = 128;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, parallel_blocks, use_logit_softcap>;
            launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 32;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int parallel_blocks = 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
    }
}

#endif // GGML_MINIMIZE_CODE_SIZE

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP getrows.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP getrows.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10, const int64_t ne11,*/ const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    const int i00 = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    const int i10 =  blockDim.y*blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib   =  i00/qk;      // block index
    const int iqs  = (i00%qk)/qr;  // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0]        = v.x;
    dst_row[iybs + iqs + y_offset] = v.y;
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10, const int64_t ne11,*/ const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    const int i00 =  blockIdx.x*blockDim.x + threadIdx.x;
    const int i10 =  blockDim.y*blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst, const int64_t ncols, const int64_t nrows_grad) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int dst_row = blockIdx.y*blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += grad[i*ncols + col];
    }

    dst[dst_row*ncols + col] = sum;
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(block_num_x, ne10, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_dd, src1_dd, dst_dd,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10, ne11,*/ ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template<typename src0_t>
static void get_rows_cuda_float(
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const src0_t * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne13 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne10, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
        src0_dd, src1_dd, dst_dd,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10, ne11,*/ ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const void    * src0_d = (const void    *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float(src0, src1, dst, (const half *) src0_d, src1_d, dst_d, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float(src0, src1, dst, (const float *) src0_d, src1_d, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, src0_d, src1_d, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, src0_d, src1_d, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, src0_d, src1_d, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, src0_d, src1_d, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_d, dst_d, stream);
            break;
        default:
            // TODO: k-quants
            GGML_ABORT("%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            break;
    }
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne1, 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP im2col.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP im2col.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_IM2COL_BLOCK_SIZE 256

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst, int64_t batch_offset,
        int64_t offset_delta, int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH, int64_t pelements, int64_t CHW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= pelements) {
        return;
    }

    const int64_t  ksize = OW * (KH > 1 ? KW : 1);
    const int64_t  kx = i / ksize;
    const int64_t  kd = kx * ksize;
    const int64_t  ky = (i - kd) / OW;
    const int64_t  ix = i % OW;

    const int64_t  oh = blockIdx.y;
    const int64_t  batch = blockIdx.z / IC;
    const int64_t  ic = blockIdx.z % IC;

    const int64_t iiw = ix * s0 + kx * d0 - p0;
    const int64_t iih = oh * s1 + ky * d1 - p1;

    const int64_t offset_dst =
        ((batch * OH + oh) * OW + ix) * CHW +
        (ic * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = ic * offset_delta + batch * batch_offset;
        dst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int parallel_elements = OW * KW * KH;
    const int num_blocks = (parallel_elements + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, OH, batch * IC);
    im2col_kernel<<<block_nums, CUDA_IM2COL_BLOCK_SIZE, 0, stream>>>(x, dst, batch_offset, offset_delta, IC, IW, IH, OH, OW, KW, KH, parallel_elements, (IC * KH * KW), s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const size_t  delta_offset = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t batch        = src1->ne[is_2D ? 3 : 2];
    const size_t  batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mmq.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mmq.cuh
//
////////////////////////////////////////////////////////////////////////////////




#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K 256
#define MMQ_NWARPS 8

typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int & kbx0, const int & i_max, const int & stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00);
typedef void (*mmq_write_back_t)(const float * __restrict__ sum, float * __restrict__ dst, const int & stride, const int & i_max, const int & j_max);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

static int get_mmq_x_max_host(const int cc) {
    return new_mma_available(cc) ? 128 :
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_OFFSET_AMD ?
#ifdef GGML_CUDA_FORCE_MMQ
            128                     : 64;
#else
            MMQ_DP4A_MAX_BATCH_SIZE : 64;
#endif // GGML_CUDA_FORCE_MMQ
}

static constexpr __device__ int get_mmq_x_max_device() {
#ifdef NEW_MMA_AVAILABLE
    return 128;
#else // NEW_MMA_AVAILABLE

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    return 128;
#else // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#ifdef GGML_CUDA_FORCE_MMQ
    return MMQ_DP4A_MAX_BATCH_SIZE;
#else // GGML_CUDA_FORCE_MMQ
    return 128;
#endif // GGML_CUDA_FORCE_MMQ
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

    return 64;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#endif // NEW_MMA_AVAILABLE
}

static int get_mmq_y_host(const int cc) {
    return cc >= GGML_CUDA_CC_OFFSET_AMD ? (GGML_CUDA_CC_IS_RDNA1(cc) ? 64 : 128) :
        (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA ? 128 : 64);
}

static constexpr __device__ int get_mmq_y_device() {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA1)
    return 64;
#else
    return 128;
#endif // defined RDNA1
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 128;
#else
    return 64;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
}

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_0   + mmq_y/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_1   + mmq_y/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*2/QI8_0 + mmq_y/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*4/QI8_0 + mmq_y/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*2/QI8_1 + mmq_y/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE         + mmq_y,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y,                                     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_K,                     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_K   + mmq_y/QI5_K,     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI6_K   + mmq_y/QI6_K,     mmq_y*WARP_SIZE/8 + mmq_y/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    return type == GGML_TYPE_Q4_0 ? MMQ_DP4A_TXS_Q4_0 :
        type == GGML_TYPE_Q4_1    ? MMQ_DP4A_TXS_Q4_1 :
        type == GGML_TYPE_Q5_0    ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_Q5_1    ? MMQ_DP4A_TXS_Q8_1 :
        type == GGML_TYPE_Q8_0    ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_Q2_K    ? MMQ_DP4A_TXS_Q2_K :
        type == GGML_TYPE_Q3_K    ? MMQ_DP4A_TXS_Q3_K :
        type == GGML_TYPE_Q4_K    ? MMQ_DP4A_TXS_Q4_K :
        type == GGML_TYPE_Q5_K    ? MMQ_DP4A_TXS_Q5_K :
        type == GGML_TYPE_Q6_K    ? MMQ_DP4A_TXS_Q6_K :
        type == GGML_TYPE_IQ2_XXS ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ2_XS  ? MMQ_DP4A_TXS_Q8_0_16 :
        type == GGML_TYPE_IQ2_S   ? MMQ_DP4A_TXS_Q8_0_16 :
        type == GGML_TYPE_IQ3_XXS ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ3_S   ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ1_S   ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ4_XS  ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ4_NL  ? MMQ_DP4A_TXS_Q8_0 :
        tile_x_sizes{0, 0, 0};
}

#define MMQ_MMA_TILE_X_K_Q8_0 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q8_1 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q2_K (2*WARP_SIZE + WARP_SIZE                         + 4)
#define MMQ_MMA_TILE_X_K_Q3_K (2*WARP_SIZE + WARP_SIZE/2                       + 4)
#define MMQ_MMA_TILE_X_K_Q6_K (2*WARP_SIZE + WARP_SIZE/QI6_K     + WARP_SIZE/8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q6_K % 8 == 4, "Wrong padding.");

static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q4_1    ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q5_0    ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q5_1    ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q8_0    ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q2_K    ? MMQ_MMA_TILE_X_K_Q2_K :
        type == GGML_TYPE_Q3_K    ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_Q4_K    ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q5_K    ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q6_K    ? MMQ_MMA_TILE_X_K_Q6_K :
        type == GGML_TYPE_IQ2_XXS ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ2_XS  ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_IQ2_S   ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_IQ3_XXS ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ3_S   ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ1_S   ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ4_XS  ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ4_NL  ? MMQ_MMA_TILE_X_K_Q8_0 :
        0;
}

#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)

static int mmq_get_granularity_host(const int mmq_x, const int cc) {
    return new_mma_available(cc) && mmq_x >= 48 ? 16 : 8;
}

#ifdef NEW_MMA_AVAILABLE
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}
#else
static constexpr __device__ int mmq_get_granularity_device(const int /* mmq_x */) {
    return 8;
}
#endif // NEW_MMA_AVAILABLE

// ------------------------------------------------------------

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
                for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + QI4_0)];
                }

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i*(WARP_SIZE + 1) + k0/QR4_0], u,
                     x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_1;
    const int kqsx = threadIdx.x % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI4_1) + kqsx + 0]     = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + threadIdx.y * QI4_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + kbxd] = bxi->dm;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_1*VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
                for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + QI4_1)];
                }

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i*(WARP_SIZE + 1) + k0/QR4_1], u,
                     x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI5_0;
    const int kqsx = threadIdx.x % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + threadIdx.y * QI5_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI5_0) + i/QI5_0 + kbxd] = bxi->d;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI5_1;
    const int kqsx = threadIdx.x % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*(2*WARP_SIZE + 1)     + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + threadIdx.y * QI5_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1       + kbxd] = bxi->dm;
#else
        x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + kbxd] = bxi->dm;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI8_0;
    const int kqsx = threadIdx.x % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 0         + threadIdx.x] = get_int_b2(bxi[0].qs,               kqsx);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE/QI8_0].qs, kqsx);
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 0         + threadIdx.x] = get_int_b2(bxi[0].qs,               kqsx);
        x_qs[i*(2*WARP_SIZE + 1)     + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE/QI8_0].qs, kqsx);
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = 2*WARP_SIZE / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0/2) {
        int i = i0 + threadIdx.y * (QI8_0/2) + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0             + kbxd] = bxi->d;
#else
        x_df[i*(2*WARP_SIZE/QI8_0) + i/(QI8_0/2) + kbxd] = bxi->d;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % WARP_SIZE],
                     x_df[i*(2*WARP_SIZE/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (WARP_SIZE/QI8_1)]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    typedef mma_A_I16K8<int> mma_A;
    typedef mma_B_J8K8<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    mma_A A[ntx][WARP_SIZE/QI8_0];
    float dA[ntx][mma_C::ne/2][WARP_SIZE/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
            const int k0 = k00 + k01;

            A[n][k01/QI8_0].load_ldmatrix(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
            mma_B  B;
            float dB[mma_C::ne/2];

            B.load_generic(y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma(A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    typedef mma_A_I16K8<int> mma_A;
    typedef mma_B_J8K8<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    mma_A    A[ntx][WARP_SIZE/QI8_1];
    float2 dmA[ntx][mma_C::ne/2][WARP_SIZE/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            A[n][k01/QI8_1].load_ldmatrix(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_A::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B    B;
            float2 dsB[mma_C::ne/2];

            B.load_generic(y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma(A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i*(2*WARP_SIZE + 1) + k0],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef NEW_MMA_AVAILABLE

    typedef mma_A_I16K4<int> mma_A;
    typedef mma_A_I16K8<int> mma_A_K8;
    typedef mma_B_J8K4<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            ((mma_A_K8 *) A[n])[k01/8].load_ldmatrix(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            mma_B B[2];
            float dB[mma_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            B[0].load_generic(y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),        MMQ_TILE_Y_K);
            B[1].load_generic(y_qs + j0*MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma(A[n][k01/4 + 0], B[0]);
                C[1].mma(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI2_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/QI2_K) {
        int i = i0 + threadIdx.y*(WARP_SIZE/QI2_K) + threadIdx.x/QI2_K;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k] = x_qs_k;
#endif // NEW_MMA_AVAILABLE
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#ifdef NEW_MMA_AVAILABLE
        x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
#else
        x_dm[i*(WARP_SIZE + 1)       + kqsx] = x_dm_ik;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    float2 y_df[mmq_x/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                if (k01 < WARP_SIZE/2) {
                    constexpr int ns = 2;
                    sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                        &x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                        &x_dm[i*(WARP_SIZE + 1) + k0/4], k01 < WARP_SIZE/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                        &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
                } else {
                    constexpr int ns = 1;
                    sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                        &x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                        &x_dm[i*(WARP_SIZE + 1) + k0/4], k01 < WARP_SIZE/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                        &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef NEW_MMA_AVAILABLE

    typedef mma_A_I16K4<int> mma_A;
    typedef mma_A_I16K8<int> mma_A_K8;
    typedef mma_B_J8K4<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + WARP_SIZE*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne/2][8];
    float  mA[ntx][mma_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            ((mma_A_K8 *) A[n])[k01/QI8_1].load_ldmatrix(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float2 dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            B[0].load_generic(y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),        MMQ_TILE_Y_K);
            B[1].load_generic(y_qs + j0*MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

            mma_C Cm[2];
            if (k01 >= WARP_SIZE * 3/4) {
                mma_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                Cm[0].mma(A1, B[0]);
                Cm[1].mma(A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C Cd[2];

                Cd[0].mma(A[n][k01/4 + 0], B[0]);
                Cd[1].mma(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= WARP_SIZE * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += tmp*(k01 < WARP_SIZE/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE * 3/4; k01 += QI8_1) {
            float2 sB[mma_C::ne/2];

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI3_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/QI3_K) {
        int i = i0 + threadIdx.y * (WARP_SIZE/QI3_K) + threadIdx.x / QI3_K;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k] = x_qs_k;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k] = x_qs_k;
#endif // NEW_MMA_AVAILABLE
        }
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*8) {
        int i = i0 + threadIdx.y*8 + threadIdx.x/(WARP_SIZE/8);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % (WARP_SIZE/8);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

#ifdef NEW_MMA_AVAILABLE
        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + sizeof(int)*(threadIdx.x % (WARP_SIZE/8)) + l] = d*sc8[l];
        }
#else
        x_sc[i*(WARP_SIZE/8) + i/8 + threadIdx.x % (WARP_SIZE/8)] = sc;
#endif // NEW_MMA_AVAILABLE
    }

#ifndef NEW_MMA_AVAILABLE
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*WARP_SIZE) {
        int i = (i0 + threadIdx.y*WARP_SIZE + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        x_df[i] = bxi->d;
    }
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int8_t * scales = ((const int8_t *) (x_sc + i*(WARP_SIZE/8) + i/8)) + k0/4;

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i*(2*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, threadIdx.x);

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(threadIdx.x/8) + threadIdx.x % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(threadIdx.x/8) + threadIdx.x % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // NEW_MMA_AVAILABLE
    }

#ifdef NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*16) {
        int i = (i0 + threadIdx.y*16 + threadIdx.x/(WARP_SIZE/16)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE/16);

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
        }
    }

#else

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*QI4_K) {
        int i = (i0 + threadIdx.y*QI4_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(WARP_SIZE/8) + i/8 + ksc] = scales8;
    }
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/32] + 2*(k01/16);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(WARP_SIZE + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
        const int ky = QR5_K*threadIdx.x;

        const int ql = get_int_b4(bxi->qs, threadIdx.x);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, threadIdx.x % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (threadIdx.x / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (threadIdx.x / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + QI5_K/4;

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kq0] = ql0 | qh0;
        x_qs[i*(2*WARP_SIZE + 1)     + kq1] = ql1 | qh1;
#endif // NEW_MMA_AVAILABLE
    }

#ifdef NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*16) {
        int i = (i0 + threadIdx.y*16 + threadIdx.x/(WARP_SIZE/16)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE/16);

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
        }
    }

#else

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*QI5_K) {
        int i = (i0 + threadIdx.y*QI5_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*8) {
        int i = (i0 + threadIdx.y*8 + threadIdx.x/(WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(WARP_SIZE/8) + i/8 + ksc] = scales8;
    }
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR5_K*VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k00/32]) + 2*(k01/16);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i*(QR5_K*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
    int   * x_sc = (int   *) (x_df + WARP_SIZE/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, threadIdx.x);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (threadIdx.x / (QI6_K/2)) + threadIdx.x % (QI6_K/4));
        const int qh0 = ((qh >> ((threadIdx.x & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((threadIdx.x & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*threadIdx.x - threadIdx.x % (QI6_K/2) + 0;
        const int kq1 = 2*threadIdx.x - threadIdx.x % (QI6_K/2) + QI6_K/2;

#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*WARP_SIZE + 1)     + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + threadIdx.y * QI6_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q6_K       + kbxd] = bxi->d;
#else
        x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K + kbxd] = bxi->d;
#endif // NEW_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / 4;

#ifdef NEW_MMA_AVAILABLE
        x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + threadIdx.x % (WARP_SIZE/8)] = get_int_b2(bxi->scales, threadIdx.x % (QI6_K/8));
#else
        x_sc[i*(WARP_SIZE/8) + i/8   + threadIdx.x % (WARP_SIZE/8)] = get_int_b2(bxi->scales, threadIdx.x % (QI6_K/8));
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR6_K*VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i*(QR6_K*WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                    x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef NEW_MMA_AVAILABLE

    typedef mma_A_I16K4<int> mma_A;
    typedef mma_B_J8K4<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * x_sc = (const int   *) x_df + WARP_SIZE/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    int   scA[ntx][mma_C::ne/2][8];
    float  dA[ntx][mma_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            A[n][k01/4 + 0].load_ldmatrix(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + 0),        MMQ_MMA_TILE_X_K_Q6_K);
            A[n][k01/4 + 1].load_ldmatrix(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + mma_A::K), MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

                const int      sc_packed = x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q6_K];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float tmp[ntx][mma_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            mma_B B[2];
            float dB[mma_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            B[0].load_generic(y_qs + j0*MMQ_TILE_Y_K + 0        + k01, MMQ_TILE_Y_K);
            B[1].load_generic(y_qs + j0*MMQ_TILE_Y_K + mma_B::K + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma(A[n][k01/4 + 0], B[0]);
                C[1].mma(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0/mma_C::J + n)*mma_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_nl(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = threadIdx.x / QI4_NL;
    const int kqsx = threadIdx.x % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;
#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 4] = v.y;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_NL;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_NL) {
        int i = i0 + threadIdx.y * QI4_NL + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(bxi->d);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kbxd] = __half2float(bxi->d);
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xxs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_XXS/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI2_XXS/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI2_XXS) + threadIdx.x/(QI2_XXS/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *) x + kbx0 + i*stride;

        const int q2 = get_int_b2(bxi->qs, 2*kqsx+0);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const int * grid_pos = (const int *) (iq2xxs_grid + aux8[l]);
            const int signs_packed = ksigns_iq2xs[(aux32 >> (7*l)) & 0x7F];

            const int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
            const int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);

            const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
            const int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid1;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = (ls*d + d/2)/4;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = (ls*d + d/2)/4;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_XS/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI2_XS/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI2_XS) + threadIdx.x/(QI2_XS/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint32_t * grid_pos = (const uint32_t *)(iq2xs_grid + (q2[l] & 0x000001FF));
            const uint32_t * signs    = (const uint32_t *)(ksigns64   + (q2[l] >> 9));

            const int grid_l = __vsub4(grid_pos[0] ^ signs[0], signs[0]);
            const int grid_h = __vsub4(grid_pos[1] ^ signs[1], signs[1]);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_S/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI2_S/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI2_S) + threadIdx.x/(QI2_S/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_xxs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI3_XXS/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI3_XXS/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI3_XXS) + threadIdx.x/(QI3_XXS/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);

            const int * signs = (const int *)(ksigns64 + ((aux32 >> (7*l)) & 0x7F));

            const int grid_l = __vsub4(grid_pos.x ^ signs[0], signs[0]);
            const int grid_h = __vsub4(grid_pos.y ^ signs[1], signs[1]);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = (ls*d + d/2)/2;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = (ls*d + d/2)/2;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI3_S/2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/(QI3_S/2)) {
        int i = i0 + threadIdx.y*(2*WARP_SIZE/QI3_S) + threadIdx.x/(QI3_S/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l+1)] = grid_h;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = ls*d;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = ls*d;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI1_S;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/QI1_S) {
        int i = i0 + threadIdx.y*(WARP_SIZE/QI1_S) + threadIdx.x/QI1_S;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq1_s * bxi = (const block_iq1_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

    #pragma unroll
        for (int l = 0; l < QR1_S/2; ++l) {
            const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

#ifdef NEW_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+1)] = grid1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + (2*l+1)] = grid1;
#endif // NEW_MMA_AVAILABLE
        }

        const float  d1q   = __half2float(bxi->d) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

#ifdef NEW_MMA_AVAILABLE
        x_ds[i*MMQ_MMA_TILE_X_K_Q8_1 + kqsx] = make_half2(d1q, d1q*delta);
#else
        x_ds[i*(WARP_SIZE/4) + i/4   + kqsx] = make_half2(d1q, d1q*delta);
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef NEW_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx  = 0;           // threadIdx.x / QI4_XS
    const int kqsx = threadIdx.x; // threadIdx.x % QI4_XS

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;
#ifdef NEW_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + k0 + 4] = v.y;
#endif // NEW_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

#ifdef NEW_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * (ls - 32);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * (ls - 32);
#endif // NEW_MMA_AVAILABLE
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
    const float * __restrict__ sum, float * __restrict__ dst, const int & stride, const int & i_max, const int & j_max) {

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[j*stride + i] = sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
    const float * __restrict__ sum, float * __restrict__ dst, const int & stride, const int & i_max, const int & j_max) {

    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_C::I);
#ifdef NEW_MMA_AVAILABLE
    static_assert(nwarps*mma_C::I == mmq_y, "nwarps*mma_C::I != mmq_y");
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * mma_C::J + mma_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*mma_C::I + mma_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[j*stride + i] = sum[(j0/mma_C::J + n)*mma_C::ne + l];
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

template <int mmq_x, int mmq_y, int nwarps, bool need_check, ggml_type type>
struct mmq_type_traits;

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_DS4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_0> {
    static constexpr int              vdr          = VDR_Q5_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_1> {
    static constexpr int              vdr          = VDR_Q5_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q8_0> {
    static constexpr int              vdr          = VDR_Q8_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q8_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q2_K> {
    static constexpr int              vdr          = VDR_Q2_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q3_K> {
    static constexpr int              vdr          = VDR_Q3_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr          = VDR_Q4_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_K> {
    static constexpr int              vdr          = VDR_Q5_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_K> {
    static constexpr int              vdr          = VDR_Q6_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_XXS> {
    static constexpr int              vdr          = VDR_IQ2_XXS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_xxs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_XS> {
    static constexpr int              vdr          = VDR_IQ2_XS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_xs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_S> {
    static constexpr int              vdr          = VDR_IQ2_S_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_s<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_XXS> {
    static constexpr int              vdr          = VDR_IQ3_XXS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_xxs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_S> {
    static constexpr int              vdr          = VDR_IQ3_S_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_s<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ1_S> {
    static constexpr int              vdr          = VDR_IQ1_S_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq1_s<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_NL> {
    static constexpr int              vdr          = VDR_IQ4_NL_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_nl<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_XS> {
    static constexpr int              vdr          = VDR_IQ4_XS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_xs<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <ggml_type type, int mmq_x, int nwarps, bool need_check, bool fixup>
static __device__ void mul_mat_q_process_tile(
    const char * __restrict__ x, const char * __restrict__ yc, float * __restrict__ dst, float * __restrict__ tmp_fixup,
    const int & ne00, const int & ne01, const int & stride01, const int & ne10, const int & ne11, const int & stride11, const int & ne0,
    const int & it, const int & jt, const int & kb0_start, const int & kb0_stop) {

    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::load_tiles;

    extern __shared__ char data_mul_mat_q[];
    int * tile_y = (int *) data_mul_mat_q;
    int * tile_x = tile_y + GGML_PAD(mmq_x*(WARP_SIZE + WARP_SIZE/QI8_1), nwarps*WARP_SIZE);

#ifdef NEW_MMA_AVAILABLE
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<mmq_x, mmq_y, nwarps, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, nwarps, need_check>;
#endif // NEW_MMA_AVAILABLE

    constexpr int blocks_per_iter = MMQ_ITER_K / qk;

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};

    const int tile_x_max_i = ne01 - it*mmq_y - 1;
    const int tile_y_max_j = ne11 - jt*mmq_x - 1;

    const int * y = (const int *) yc + jt*(mmq_x*sizeof(block_q8_1_mmq)/sizeof(int));

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, stride01*it*mmq_y + kb0, tile_x_max_i, stride01);

        {
            const int * by0 = y + stride11*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 0*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + stride11*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 1*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, WARP_SIZE);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, dst + jt*mmq_x*ne0 + it*mmq_y, ne0, tile_x_max_i, tile_y_max_j);
    }
}


// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int mmq_x, int nwarps, bool need_check>
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
    __launch_bounds__(WARP_SIZE*nwarps, 2)
#endif // defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __launch_bounds__(WARP_SIZE*nwarps, 1)
#else
    __launch_bounds__(WARP_SIZE*nwarps, 2)
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
static __global__ void mul_mat_q(
    const char * __restrict__ x, const char * __restrict__ yc, float * __restrict__ dst, float * __restrict__ tmp_fixup,
    const int ne00, const int ne01, const int stride01, const int ne10, const int ne11, const int stride11, const int ne0) {

    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int qk    = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y = get_mmq_y_device();

    // On AMD or old CUDA the performance with stream-k was worse, use conventional tiling instead:
#if (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    {
        constexpr bool fixup = false;
        mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>
            (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
                blockIdx.x, blockIdx.y, 0, ne00/qk);
        return;
    }
#endif // (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA

    const     int64_t blocks_per_ne00 = ne00 / qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;

    const int ntx = (ne11 + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (ne01 + mmq_y - 1) / mmq_y; // Number of tiles y

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc      = (int64_t) blockIdx.x     *blocks_per_ne00*ntx*nty / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*blocks_per_ne00*ntx*nty / gridDim.x;

    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        const int jt =  kbc /    (blocks_per_ne00*nty);                    // j index of current tile.
        const int it = (kbc - jt*(blocks_per_ne00*nty)) / blocks_per_ne00; // i index of current tile.

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>
            (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
             it, jt, kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int jt =  kbc /    (blocks_per_ne00*nty);
    const int it = (kbc - jt*(blocks_per_ne00*nty)) / blocks_per_ne00;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>
        (x, yc, dst, tmp_fixup, ne00, ne01, stride01, ne10, ne11, stride11, ne0,
            it, jt, kb0_start, kb0_stop);
}


template <ggml_type type, int mmq_x, int nwarps, bool need_check>
static __global__ void mul_mat_q_stream_k_fixup(
    float * __restrict__ dst, const float * __restrict__ tmp_last_tile, const int ne00, const int ne01, const int ne11, const int ne0, const int block_num_mmq) {

    constexpr int     mmq_y           = get_mmq_y_device();
    constexpr int     qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;
    const     int64_t blocks_per_ne00 = ne00 / qk;

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};

    const int ntx = (ne11 + mmq_x - 1) / mmq_x;
    const int nty = (ne01 + mmq_y - 1) / mmq_y;

    bool any_fixup = false;

    const int bidx_start = ((blockIdx.y*nty + blockIdx.x)     * block_num_mmq)                           / (gridDim.y*gridDim.x);
    const int bidx_stop  = ((blockIdx.y*nty + blockIdx.x + 1) * block_num_mmq + gridDim.y*gridDim.x - 1) / (gridDim.y*gridDim.x);

    int64_t kbc_0;
    int64_t kbc_stop_0 = (int64_t) bidx_start*blocks_per_ne00*ntx*nty / block_num_mmq;

    for (int bidx = bidx_start; bidx < bidx_stop; ++bidx) {
        kbc_0 = kbc_stop_0;
        kbc_stop_0 = (int64_t) (bidx + 1)*blocks_per_ne00*ntx*nty / block_num_mmq;

        const int64_t kbc      = kbc_0      - (kbc_0      % blocks_per_ne00) % blocks_per_iter;
        const int64_t kbc_stop = kbc_stop_0 - (kbc_stop_0 % blocks_per_ne00) % blocks_per_iter;

        // Skip fixup tile if the MMQ CUDA block never wrote anything to it:
        if (kbc == kbc_stop || kbc_stop % blocks_per_ne00 == 0) {
            continue;
        }

        const int jt =  kbc_stop /    (blocks_per_ne00*nty);
        const int it = (kbc_stop - jt*(blocks_per_ne00*nty)) / blocks_per_ne00;

        // Skip fixup tile if it's unrelated to the output tile assigned to this CUDA block:
        if (it != blockIdx.x || jt != blockIdx.y) {
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
            }
        }
    }

    if (!any_fixup) {
        return;
    }

    dst += blockIdx.y*mmq_x*ne0 + blockIdx.x*mmq_y;

    const int i_max = ne01 - blockIdx.x*mmq_y - 1;
    const int j_max = ne11 - blockIdx.y*mmq_x - 1;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[j*ne0 + i] += sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}

struct mmq_args {
    const char * x; const char * y; float * dst;
    int64_t ne00; int64_t ne01; int64_t stride01;
    int64_t ne10; int64_t ne11; int64_t stride11;
    int64_t ne0;
    bool use_stream_k;
};

template<ggml_type type>
static int mmq_get_shmem(const int mmq_x, const int mmq_y, const int cc) {
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const int shmem_x = new_mma_available(cc) ? mmq_y*mmq_tile_x_k*sizeof(int) : txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);
    const int shmem_y = mmq_x*sizeof(block_q8_1_mmq);
    return shmem_x + GGML_PAD(shmem_y, MMQ_NWARPS*WARP_SIZE*sizeof(int));
}

template <ggml_type type, int mmq_x>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int mmq_y = get_mmq_y_host(cc);

    const dim3 block_dims(WARP_SIZE, MMQ_NWARPS, 1);

    const int shmem = mmq_get_shmem<type>(mmq_x, mmq_y, cc);

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
    static bool shmem_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shmem_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<type, mmq_x, MMQ_NWARPS, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<type, mmq_x, MMQ_NWARPS, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        shmem_limit_raised[id] = true;
    }
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))

    const int nty = (args.ne01 + mmq_y - 1) / mmq_y;
    const int ntx = (args.ne11 + mmq_x - 1) / mmq_x;
    const dim3 block_nums_xy_tiling(nty, ntx, 1);

    if (!args.use_stream_k) {
        if (args.ne01 % mmq_y == 0) {
            constexpr bool need_check = false;
            mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, shmem, stream>>>
                (args.x, args.y, args.dst, nullptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        } else {
            constexpr bool need_check = true;
            mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, shmem, stream>>>
                (args.x, args.y, args.dst, nullptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
        }
        return;
    }

    const dim3 block_nums_mmq(nsm, 1, 1);

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool, block_nums_mmq.x * mmq_x*mmq_y);

    if (args.ne01 % mmq_y == 0) {
        constexpr bool need_check = false;

        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_mmq, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);

        mul_mat_q_stream_k_fixup<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, 0, stream>>>
            (args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.ne11, args.ne0, block_nums_mmq.x);
    } else {
        constexpr bool need_check = true;

        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_mmq, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);

        mul_mat_q_stream_k_fixup<type, mmq_x, MMQ_NWARPS, need_check><<<block_nums_xy_tiling, block_dims, 0, stream>>>
            (args.dst, tmp_fixup.ptr, args.ne00, args.ne01, args.ne11, args.ne0, block_nums_mmq.x);
    }
}

template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id    = ggml_cuda_get_device();
    const int nsm   = ggml_cuda_info().devices[id].nsm;
    const int cc    = ggml_cuda_info().devices[id].cc;
    const int smpbo = ggml_cuda_info().devices[id].smpbo;

    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc);
    const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y;
    const bool use_stream_k = ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_OFFSET_AMD;

    int mmq_x_best  = 0;
    int nparts_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && nparts_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);

        if (mmq_x % granularity != 0 || mmq_get_shmem<type>(mmq_x, mmq_y, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ne11 + mmq_x - 1) / mmq_x;
        const int nwaves_xy_tiling = ntiles_x*block_num_y;
        const int nparts = use_stream_k ? ntiles_x : nwaves_xy_tiling;

        if (nparts < nparts_best) {
            mmq_x_best  = mmq_x;
            nparts_best = nparts;
        }
    }

    switch (mmq_x_best) {
        case   8:
            launch_mul_mat_q<type,   8>(ctx, args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16>(ctx, args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24>(ctx, args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32>(ctx, args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40>(ctx, args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48>(ctx, args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56>(ctx, args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64>(ctx, args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72>(ctx, args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80>(ctx, args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88>(ctx, args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96>(ctx, args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104>(ctx, args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112>(ctx, args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120>(ctx, args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128>(ctx, args, stream);
            break;
        default:
            fprintf(stderr, "mmq_x_best=%d\n", mmq_x_best);
            GGML_ABORT("fatal error");
            break;
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);
#ifndef GGML_NO_IQUANTS
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
#endif // GGML_NO_IQUANTS

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11);

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride00 = ne00 / ggml_blck_size(src0->type);

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA &&
        cc < GGML_CUDA_CC_OFFSET_AMD && src1_ncols == ne11;
    const mmq_args args = {src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stride00, src1_padded_row_size, src1_ncols, ne11, nrows_dst, use_stream_k};

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
#ifndef GGML_NO_IQUANTS
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
#endif // GGML_NO_IQUANTS
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (new_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (cc < GGML_CUDA_CC_OFFSET_AMD) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    return (!GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc) && !GGML_CUDA_CC_IS_GCN(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mmvq.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP mmvq.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? vec_dot_q4_0_q8_1 :
        type == GGML_TYPE_Q4_1 ? vec_dot_q4_1_q8_1 :
        type == GGML_TYPE_Q5_0 ? vec_dot_q5_0_q8_1 :
        type == GGML_TYPE_Q5_1 ? vec_dot_q5_1_q8_1 :
        type == GGML_TYPE_Q8_0 ? vec_dot_q8_0_q8_1 :
        type == GGML_TYPE_Q2_K ? vec_dot_q2_K_q8_1 :
        type == GGML_TYPE_Q3_K ? vec_dot_q3_K_q8_1 :
        type == GGML_TYPE_Q4_K ? vec_dot_q4_K_q8_1 :
        type == GGML_TYPE_Q5_K ? vec_dot_q5_K_q8_1 :
        type == GGML_TYPE_Q6_K ? vec_dot_q6_K_q8_1 :
        type == GGML_TYPE_IQ2_XXS ? vec_dot_iq2_xxs_q8_1 :
        type == GGML_TYPE_IQ2_XS ? vec_dot_iq2_xs_q8_1 :
        type == GGML_TYPE_IQ2_S ? vec_dot_iq2_s_q8_1 :
        type == GGML_TYPE_IQ3_XXS ? vec_dot_iq3_xxs_q8_1 :
        type == GGML_TYPE_IQ1_S ? vec_dot_iq1_s_q8_1 :
        type == GGML_TYPE_IQ1_M ? vec_dot_iq1_m_q8_1 :
        type == GGML_TYPE_IQ4_NL ? vec_dot_iq4_nl_q8_1 :
        type == GGML_TYPE_IQ4_XS ? vec_dot_iq4_xs_q8_1 :
        type == GGML_TYPE_IQ3_S ? vec_dot_iq3_s_q8_1 :
        nullptr;
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? VDR_Q4_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q4_1    ? VDR_Q4_1_Q8_1_MMVQ :
        type == GGML_TYPE_Q5_0    ? VDR_Q5_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q5_1    ? VDR_Q5_1_Q8_1_MMVQ :
        type == GGML_TYPE_Q8_0    ? VDR_Q8_0_Q8_1_MMVQ :
        type == GGML_TYPE_Q2_K    ? VDR_Q2_K_Q8_1_MMVQ :
        type == GGML_TYPE_Q3_K    ? VDR_Q3_K_Q8_1_MMVQ :
        type == GGML_TYPE_Q4_K    ? VDR_Q4_K_Q8_1_MMVQ :
        type == GGML_TYPE_Q5_K    ? VDR_Q5_K_Q8_1_MMVQ :
        type == GGML_TYPE_Q6_K    ? VDR_Q6_K_Q8_1_MMVQ :
        type == GGML_TYPE_IQ2_XXS ? VDR_IQ2_XXS_Q8_1_MMVQ :
        type == GGML_TYPE_IQ2_XS  ? VDR_IQ2_XS_Q8_1_MMVQ :
        type == GGML_TYPE_IQ2_S   ? VDR_IQ2_S_Q8_1_MMVQ :
        type == GGML_TYPE_IQ3_XXS ? VDR_IQ3_XXS_Q8_1_MMVQ :
        type == GGML_TYPE_IQ3_S   ? VDR_IQ3_S_Q8_1_MMVQ :
        type == GGML_TYPE_IQ4_NL  ? VDR_IQ4_NL_Q8_1_MMVQ :
        type == GGML_TYPE_IQ4_XS  ? VDR_IQ4_XS_Q8_1_MMVQ :
        1;
}

template <ggml_type type, int ncols_y>
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__((ncols_y <= 4 ? 4 : 2)*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int nwarps              = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <ggml_type type>
static void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = 1;

    if (ggml_cuda_info().devices[id].cc < GGML_CUDA_CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        switch(ncols_y) {
            case 1:
                nwarps = 4;
                rows_per_cuda_block = 1;
                break;
            case 2:
            case 3:
            case 4:
                nwarps = 4;
                rows_per_cuda_block = 2;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                nwarps = 2;
                rows_per_cuda_block = 2;
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
    }

    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<type, 1><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 2:
            mul_mat_vec_q<type, 2><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 3:
            mul_mat_vec_q<type, 3><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 4:
            mul_mat_vec_q<type, 4><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 5:
            mul_mat_vec_q<type, 5><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 6:
            mul_mat_vec_q<type, 6><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 7:
            mul_mat_vec_q<type, 7><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 8:
            mul_mat_vec_q<type, 8><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void mul_mat_vec_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_0>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_0>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q8_0>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q2_K>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q3_K>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_K>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_K>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q6_K>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_XXS>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_XS>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_S>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_XXS>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ1_S>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ1_M>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ4_NL>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ4_XS>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_S>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
#ifndef GGML_NO_IQUANTS
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_iq2_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_iq2_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_iq2_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_iq3_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_iq1_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_iq1_m_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_iq4_nl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_iq4_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_iq3_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
#endif // GGML_NO_IQUANTS
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP norm.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP norm.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float2 s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size>
static __global__ void rms_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP pad.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP pad.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_PAD_BLOCK_SIZE 256

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void pad_f32(const float * x, float * dst, const int ne0, const int ne00, const int ne01, const int ne02, const int ne03) {
    // blockIdx.z: idx of ne2*ne3, aka ne02*ne03
    // blockIdx.y: idx of ne1
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    if (nidx < ne00 && blockIdx.y < ne01 && blockIdx.z < ne02*ne03) {
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}

static void pad_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int ne0, const int ne1, const int ne2, const int ne3, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    pad_f32_cuda(src0_d, dst_d,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP pool2d.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP pool2d.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_POOL2D_BLOCK_SIZE 256

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <typename Ti, typename To>
static  __global__ void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int I_HW = ih * iw;
    const int O_HW = oh * ow;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / ow;
    const int cur_ow = idx % O_HW % ow;
    const Ti* i_ptr = src + nc * I_HW;
    To* o_ptr = dst + nc * O_HW;
    const int start_h = cur_oh * sh - ph;
    const int bh = max(0, start_h);
    const int eh = min(ih, start_h + kh);
    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);
    const To scale = 1. / (kh * kw);
    To res = 0;

    switch (op) {
        case GGML_OP_POOL_AVG: res = 0; break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: assert(false);
    }

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
#if __CUDA_ARCH__ >= 350
            Ti cur = __ldg(i_ptr + i * iw + j);
#else
            Ti cur = i_ptr[i * iw + j];
#endif
            switch (op) {
                case GGML_OP_POOL_AVG: res += cur * scale; break;
                case GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
                default: assert(false);
            }
        }
    }
    o_ptr[cur_oh * ow + cur_ow] = res;
}

static void pool2d_nchw_kernel_f32_f32_cuda(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {

    const int num_blocks = (parallel_elements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool2d_nchw_kernel<<<block_nums, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = src0->ne[1];
    const int64_t IW = src0->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;

    pool2d_nchw_kernel_f32_f32_cuda(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0, parallel_elements, src0_d, dst_d, op, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP quantize.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP quantize.cuh
//
////////////////////////////////////////////////////////////////////////////////




#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx0_padded) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix0 < kx ? x[ix1*kx + ix0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
    const float * __restrict__ x, void * __restrict__ vy, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t ix0 = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*4;

    if (ix0 >= kx0_padded) {
        return;
    }

    const float4 * x4 = (const float4 *) x;

    const int64_t ix1 = kx1*blockIdx.z + blockIdx.y;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.y*gridDim.x*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (ix0 / (4*QK8_1))*kx1 + blockIdx.y;                   // block index in channel
    const int64_t iqs = ix0 % (4*QK8_1);                                            // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = ix0 < kx0 ? x4[(ix1*kx0 + ix0)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Exchange calculate sum across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % QK8_1 == 0);

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1*channels, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx0_padded);

    GGML_UNUSED(type_x);
}

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % (4*QK8_1) == 0);

    const int64_t block_num_x = (kx0_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(block_num_x, kx1, channels);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_x)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP rope.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP rope.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_norm(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0;
    const int ix   = channel_x*s2 + row_x*s1 + i0;

    const float theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    dst[idst + 0] = x0*cos_theta - x1*sin_theta;
    dst[idst + 1] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_neox(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const float theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_multi(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2,
        const int n_dims, const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors, const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        theta_base = pos[channel_x + ne2 * 1]*powf(theta_scale, i0/2.0f);
    }
    else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 2]*powf(theta_scale, i0/2.0f);
    }
    else if (sector >= sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 3]*powf(theta_scale, i0/2.0f);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_vision(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
        const float theta_scale, const float * freq_factors, const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base = pos[channel_x]*powf(theta_scale, p);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base = pos[channel_x + ne2]*powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, typename T>
static void rope_norm_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
}

template<bool forward, typename T>
static void rope_neox_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    } else {
        rope_neox<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
}

template<bool forward, typename T>
static void rope_multi_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, const mrope_sections sections, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template<bool forward, typename T>
static void rope_vision_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, const mrope_sections sections, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    mrope_sections sections;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32) {
            rope_neox_cuda<forward>(
                (const float *) src0_d, (float *) dst_d, ne00, ne01, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_neox_cuda<forward>(
                (const half *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>(
                (const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>(
                (const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>(
                (const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>(
                (const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32) {
            rope_norm_cuda<forward>(
                (const float *) src0_d, (float *) dst_d, ne00, ne01, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_norm_cuda<forward>(
                (const half *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP scale.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP scale.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void scale_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, ggml_nelements(src0), stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP softmax.cu
//
////////////////////////////////////////////////////////////////////////////////

#include "ggml.h"

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP softmax.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(
        const float * x, const T * mask, float * dst, const int ncols_par, const int nrows_y,
        const float scale, const float max_bias, const float m0, const float m1, uint32_t n_head_log2) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    x    += int64_t(rowx)*ncols;
    mask += int64_t(rowy)*ncols * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(max_bias, rowx/nrows_y, n_head_log2, m0, m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static __global__ void soft_max_back_f32(
        const float * grad, const float * dstf, float * dst, const int ncols, const float scale) {
    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template<typename T>
static void soft_max_f32_cuda(const float * x, const T * mask, float * dst, const int ncols_x, const int nrows_x, const int nrows_y, const float scale, const float max_bias, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t nbytes_shared = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    const uint32_t n_head      = nrows_x/nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    if (nbytes_shared < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        switch (ncols_x) {
            case 32:
                soft_max_f32<true,   32,   32><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 64:
                soft_max_f32<true,   64,   64><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 128:
                soft_max_f32<true,  128,  128><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 256:
                soft_max_f32<true,  256,  256><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 512:
                soft_max_f32<true,  512,  512><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 1024:
                soft_max_f32<true, 1024, 1024><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 2048:
                soft_max_f32<true, 2048, 1024><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 4096:
                soft_max_f32<true, 4096, 1024><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            default:
                soft_max_f32<true,    0,    0><<<block_nums, block_dims, nbytes_shared, stream>>>
                    (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
        }
    } else {
        const size_t nbytes_shared_low = WARP_SIZE*sizeof(float);
        soft_max_f32<false, 0, 0><<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
    }
}

static void soft_max_back_f32_cuda(
        const float * grad, const float * dstf, float * dst,
        const int ncols, const int nrows, const float scale, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows,     1, 1);

    soft_max_back_f32<<<block_nums, block_dims, 0, stream>>>(grad, dstf, dst, ncols, scale);
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    if (use_f16) {
        soft_max_f32_cuda(src0_d, (const half  *) src1_d, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    } else {
        soft_max_f32_cuda(src0_d, (const float *) src1_d, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    }
}

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_cuda(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP sumrows.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP sumrows.cuh
//
////////////////////////////////////////////////////////////////////////////////


void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void k_sum_rows_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col == 0) {
        dst[row] = sum;
    }
}

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    k_sum_rows_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP tsembd.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP tsembd.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void timestep_embedding_f32(const float * timesteps, float * dst, const int nb1, const int dim, const int max_period) {
    // blockIDx.y: idx of timesteps->ne[0]
    // blockIDx.x: idx of ((dim + 1) / 2) / BLOCK_SIZE
    int i = blockIdx.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    float * embed_data = (float *)((char *)dst +  i*nb1);

    if (dim % 2 != 0 && j == ((dim + 1) / 2)) {
        embed_data[dim] = 0.f;
    }

    int half = dim / 2;
    if (j >= half) {
        return;
    }

    float timestep = timesteps[i];
    float freq = (float)expf(-logf(max_period) * j / half);
    float arg = timestep * freq;
    embed_data[j] = cosf(arg);
    embed_data[j + half] = sinf(arg);
}

static void timestep_embedding_f32_cuda(const float * x, float * dst, const int ne00, const int nb1,
                                        const int dim, const int max_period, cudaStream_t stream) {
    int half_ceil = (dim + 1) / 2;
    int num_blocks = (half_ceil + CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne00, 1);
    timestep_embedding_f32<<<gridDim, CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE, 0, stream>>>(x, dst, nb1, dim, max_period);
}

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];

    timestep_embedding_f32_cuda(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP unary.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP unary.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void neg_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = -x[i];
}

static __global__ void step_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] > 0.0f;
}

static __global__ void gelu_f32(const float * x, float * dst, const int k) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f*xi*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void gelu_quick_f32(const float * x, float * dst, int k) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x[i])));
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void silu_back_f32(
        const float * grad, const float * xf, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xfi = xf[i];
    const float s = 1.0f / (1.0f + expf(-xfi));
    dst[i] = grad[i] * s * (1.0f + xfi * (1.0f - s));
}

static __global__ void tanh_f32(const float * x, float * dst, int k) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = tanhf(x[i]);
}

static __global__ void relu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

static __global__ void sigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (1.0f + expf(-x[i]));
}

static __global__ void hardsigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void hardswish_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void exp_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = expf(x[i]);
}

static __global__ void leaky_relu_f32(const float * x, float * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) + fminf(x[i], 0.0f) * negative_slope;
}

static __global__ void sqr_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static __global__ void sqrt_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sqrtf(x[i]);
}

static __global__ void sin_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sinf(x[i]);
}

static __global__ void cos_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = cosf(x[i]);
}

static void neg_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    neg_f32<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void step_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_STEP_BLOCK_SIZE - 1) / CUDA_STEP_BLOCK_SIZE;
    step_f32<<<num_blocks, CUDA_STEP_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void gelu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void gelu_quick_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_quick_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void silu_back_f32_cuda(const float * grad, const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_f32<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

static void tanh_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    tanh_f32<<<num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void sigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    sigmoid_f32<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardsigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    hardsigmoid_f32<<<num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardswish_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    hardswish_f32<<<num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void exp_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_EXP_BLOCK_SIZE - 1) / CUDA_EXP_BLOCK_SIZE;
    exp_f32<<<num_blocks, CUDA_EXP_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void leaky_relu_f32_cuda(const float * x, float * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

static void sqr_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQR_BLOCK_SIZE - 1) / CUDA_SQR_BLOCK_SIZE;
    sqr_f32<<<num_blocks, CUDA_SQR_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void sqrt_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQRT_BLOCK_SIZE - 1) / CUDA_SQRT_BLOCK_SIZE;
    sqrt_f32<<<num_blocks, CUDA_SQRT_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void sin_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIN_BLOCK_SIZE - 1) / CUDA_SIN_BLOCK_SIZE;
    sin_f32<<<num_blocks, CUDA_SIN_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void cos_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_COS_BLOCK_SIZE - 1) / CUDA_COS_BLOCK_SIZE;
    cos_f32<<<num_blocks, CUDA_COS_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    neg_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    step_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_back_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    tanh_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardsigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardswish_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    exp_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), negative_slope, stream);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqrt_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sin_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    cos_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP upscale.cu
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP upscale.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_UPSCALE_BLOCK_SIZE 256

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void upscale_f32(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const float sf0, const float sf1, const float sf2, const float sf3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }

    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = i10 / sf0;
    int i01 = i11 / sf1;
    int i02 = i12 / sf2;
    int i03 = i13 / sf3;

    dst[index] = *(float *)((char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00);
}

static void upscale_f32_cuda(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const float sf0, const float sf1, const float sf2, const float sf3,
        cudaStream_t stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    upscale_f32<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const float sf0 = (float)dst->ne[0]/src0->ne[0];
    const float sf1 = (float)dst->ne[1]/src0->ne[1];
    const float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    upscale_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP ggml-cuda.cu
//
////////////////////////////////////////////////////////////////////////////////

#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"


////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP argmax.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP count-equal.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_COUNT_EQUAL_CHUNK_SIZE 128

void ggml_cuda_count_equal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP cross-entropy-loss.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE 256

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP opt-step-adamw.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_OPT_STEP_ADAMW_BLOCK_SIZE 256

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP out-prod.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP sum.cuh
//
////////////////////////////////////////////////////////////////////////////////


void sum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream);

void ggml_cuda_op_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP wkv6.cuh
//
////////////////////////////////////////////////////////////////////////////////


#define CUDA_WKV_BLOCK_SIZE 64

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP gla.cuh
//
////////////////////////////////////////////////////////////////////////////////


void ggml_cuda_op_gated_linear_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#include "ggml.h"


static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

GGML_NORETURN
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
#if defined(GGML_USE_HIP) && defined(GGML_HIP_UMA)
    auto res = hipMallocManaged(ptr, size);
    if (res == hipSuccess) {
        // if error we "need" to know why...
        CUDA_CHECK(hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
    }
    return res;
#else

#if !defined(GGML_USE_HIP)
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr)
    {
        err = cudaMallocManaged(ptr, size);
    }
    else
    {
        err = cudaMalloc(ptr, size);
    }
    return err;
#else
    return cudaMalloc(ptr, size);
#endif // !defined(GGML_USE_HIP)

#endif
}

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    {
        int major_version = 0;
        size_t version_length = 0;
        if (rocblas_get_version_string_size(&version_length) == rocblas_status_success) {
            std::string version(version_length, '\0');
            if (rocblas_get_version_string(version.data(), version.size()) == rocblas_status_success) {
                version.resize(::strlen(version.c_str()));
                int parsed_value = 0;
                if (std::from_chars(version.c_str(), version.c_str() + version.length(), parsed_value).ec == std::errc()) {
                    major_version = parsed_value;
                }
            }
        }
        if (major_version < 4) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " calling rocblas_initialize as a workaround for a rocBLAS bug\n");
            rocblas_initialize();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    no\n", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: no\n", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm       = prop.multiProcessorCount;
        info.devices[id].smpb      = prop.sharedMemPerBlock;
        info.devices[id].warp_size = prop.warpSize;
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess((CUdeviceptr)((char *)(pool_addr) + pool_size), reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }

        const int cc = ggml_cuda_info().devices[id].cc;
        row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
    }
    return row_rounding;
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    int main_device;
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    std::string name;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(cudaFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};


static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, size, id));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
}

static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

static const char * ggml_backend_cuda_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::pair<int, std::array<float, GGML_CUDA_MAX_DEVICES>>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find({main_device, tensor_split_arr});
    if (it != buft_map.end()) {
        return &it->second;
    }
    auto * ctx = new ggml_backend_cuda_split_buffer_type_context{
        main_device,
        tensor_split_arr,
        GGML_CUDA_NAME + std::to_string(main_device) + "_Split",
    };

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    const bool use_fp16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT;

    if (compute_capability >= GGML_CUDA_CC_VOLTA && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        if (GGML_CUDA_CC_IS_CDNA(compute_capability)) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f16, src0_ptr,      CUDA_R_16F, ne00,
                                    src1_ptr,      CUDA_R_16F, ne10,
                        &beta_f16,  dst_f16.get(), CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
            to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        }
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);

        for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != main_device && id_other != main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
                    if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    } else {
                        // reset the error
                        (void)cudaGetLastError();
                    }
                } else {
                    cudaError_t err = cudaDeviceDisablePeerAccess(id_other);
                    if (err != cudaErrorPeerAccessNotEnabled) {
                        CUDA_CHECK(err);
                    } else {
                        // reset the error
                        (void)cudaGetLastError();
                    }
                }
            }
        }
    }

    ggml_cuda_set_device(main_device);
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;

    GGML_UNUSED(main_device);
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int64_t i02_divisor = ne12 / ne02;
    const int64_t i03_divisor = ne13 / ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));
    GGML_ASSERT(!(split && ne03 < ne13));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer.
            // This buffer needs to be cleared entirely because multiple regions will function as padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
        // TODO: remove this for MUSA once the Guilty Lockup issue is resolved
#ifndef GGML_USE_MUSA
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
#else // GGML_USE_MUSA
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
#endif // !GGML_USE_MUSA
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            const size_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12*ne13, src1_padded_col_size, src0->type, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                const size_t nbytes_src0_matrix = ne01*ne00*src0_ts / src0_bs;
                char  *  src0_dd_i =  dev[id].src0_dd + ((i03/i03_divisor)*ne02 + (i02/i02_divisor)) * nbytes_src0_matrix;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i03 % i03_divisor == 0 && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src0_dd_i, src0, i03/i03_divisor, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
                        CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                            dhf_dst_i, ctx.device, ne0*sizeof(float), dst_dd_i, id, row_diff*sizeof(float), row_diff*sizeof(float), src1_ncols, stream));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

static __global__ void k_compute_batched_ptrs(
        const half * src0_as_f16, const half * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);

    cudaStream_t main_stream = ctx.stream();

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    void * src0_ddq = src0->data;
    half * src0_f16 = (half *) src0_ddq;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    // convert src1 to fp16
    ggml_cuda_pool_alloc<half> src1_f16_alloc(ctx.pool());
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda(src1_ddf, src1_f16_alloc.get(), ne_src1, main_stream);
    }
    half * src1_f16 = src1->type == GGML_TYPE_F16 ? (half *) src1_ddf : src1_f16_alloc.get();

    ggml_cuda_pool_alloc<half> dst_f16(ctx.pool());
    char * dst_t;

    cublasComputeType_t cu_compute_type = CUBLAS_COMPUTE_16F;
    cudaDataType_t      cu_data_type    = CUDA_R_16F;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const half  alpha_f16 = 1.0f;
    const half  beta_f16  = 0.0f;

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f16;
    const void * beta  = &beta_f16;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char *) dst_f16.alloc(ne_dst);

        nbd2 /= sizeof(float) / sizeof(half);
        nbd3 /= sizeof(float) / sizeof(half);
    } else {
        dst_t = (char *) dst_ddf;

        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type    = CUDA_R_32F;

        alpha = &alpha_f32;
        beta  = &beta_f32;
    }

    if (GGML_CUDA_CC_IS_CDNA(ggml_cuda_info().devices[ctx.device].cc)) {
        cu_compute_type = CUBLAS_COMPUTE_32F;
        alpha = &alpha_f32;
        beta  = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

#if 0
    // use cublasGemmEx
    {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                int i03 = i13 / r3;
                int i02 = i12 / r2;

                CUBLAS_CHECK(
                        cublasGemmEx(g_cublas_handles[g_main_device], CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            alpha, (const char *) src0_as_f16 + i02*src0->nb[2]   + i03*src0->nb[3]  , CUDA_R_16F,   nb01/sizeof(half),
                                   (const char *) src1_as_f16 + i12*src1->nb[2]/2 + i13*src1->nb[3]/2, CUDA_R_16F,   nb11/sizeof(float),
                            beta,  (      char *)       dst_t + i12*nbd2          + i13*nbd3,          cu_data_type, ne01,
                            cu_compute_type,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
#else
#ifdef GGML_USE_MUSA
    GGML_ASSERT(false);
#else // !GGML_USE_MUSA
    if (r2 == 1 && r3 == 1 && ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1)) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const char *) src0_f16, CUDA_R_16F,   nb01/nb00, nb02/nb00,  // strideA
                       (const char *) src1_f16, CUDA_R_16F,   nb11/nb10, nb12/nb10,  // strideB
                beta,  (      char *)    dst_t, cu_data_type, ne01,       nb2/nb0,   // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        dim3 block_dims(ne13, ne12);
        k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(
                src0_f16, src1_f16, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                src1->type == GGML_TYPE_F16 ? nb12 : nb12/2,
                src1->type == GGML_TYPE_F16 ? nb13 : nb13/2,
                nbd2, nbd3,
                r2, r3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), CUDA_R_16F,   nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), CUDA_R_16F,   nb11/nb10,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type, ne01,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
#endif // GGML_USE_MUSA
#endif

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);

    bool use_mul_mat_vec   = (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % 2 == 0 && src1->ne[1] == 1;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q     = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16   = false;
    bool any_gpus_without_fp16_mma = false;

    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc              = ggml_cuda_info().devices[id].cc;
            use_mul_mat_q             = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
            any_gpus_with_slow_fp16   = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
            any_gpus_without_fp16_mma = any_gpus_without_fp16_mma || !fp16_mma_hardware_available(cc);
        }
    } else {
        const int cc              = ggml_cuda_info().devices[ctx.device].cc;
        use_mul_mat_q             = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
        any_gpus_with_slow_fp16   = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
        any_gpus_without_fp16_mma = any_gpus_without_fp16_mma || !fp16_mma_hardware_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (!split && use_mul_mat_vec && (src0->ne[1] < MMV_MAX_ROWS || any_gpus_without_fp16_mma)) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16)
               && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static __global__ void k_copy_src1_to_contiguous(const char * __restrict__ src1_original, char * __restrict__ src1_contiguous,
                                                 int * __restrict__ cur_src1_row, mmid_row_mapping * __restrict__ row_mapping,
                                                 const char * __restrict ids, int64_t i02, size_t ids_nb1, size_t ids_nb0,
                                                 int64_t ne11, int64_t ne10,
                                                 size_t nb11, size_t nb12) {
    int32_t iid1 = blockIdx.x;
    int32_t id = blockIdx.y;

    const int32_t row_id_i = *(const int32_t *) (ids + iid1*ids_nb1 + id*ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    __shared__ int src1_row;
    if (threadIdx.x == 0) {
        src1_row = atomicAdd(cur_src1_row, 1);
        row_mapping[src1_row] = {id, iid1};
    }
    __syncthreads();

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

    for (int i = threadIdx.x; i < ne10; i += blockDim.x) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) && "mul_mat_id does not support split buffers");

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                src0_row.data = src0_original + i02*nb02;
                src1_row.data = src1_original + i11*nb11 + i12*nb12;
                dst_row.data  =  dst_original + i1*nb1   + i2*nb2;

                ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
        ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;

            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            if (num_src1_rows == 0) {
                continue;
            }

            ggml_cuda_pool_alloc<int> dev_cur_src1_row(ctx.pool(), 1);
            ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            CUDA_CHECK(cudaMemsetAsync(dev_cur_src1_row.get(), 0, sizeof(int), stream));

            {
                dim3 block_dims(std::min((unsigned int)ne10, 768u));
                dim3 grid_dims(ids->ne[1], n_ids);
                k_copy_src1_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                        src1_original, src1_contiguous.get(),
                        dev_cur_src1_row.get(), dev_row_mapping.get(),
                        ids_dev, i02, ids->nb[1], ids->nb[0],
                        ne11, ne10,
                        nb11, nb12);
                CUDA_CHECK(cudaGetLastError());
            }

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);

            src1_row.ne[1] = num_src1_rows;
            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                dim3 block_dims(std::min((unsigned int)ne0, 768u));
                dim3 grid_dims(num_src1_rows);
                k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                        dst_original, dst_contiguous.get(),
                        dev_row_mapping.get(),
                        ne0,
                        nb1, nb2);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // why is this here instead of mul_mat?
    if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->buft)) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
#ifndef GGML_MINIMIZE_CODE_SIZE
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
#endif
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {

        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH
static bool check_node_graph_compatibility_and_refresh_copy_ops(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    std::vector<void *> & ggml_cuda_cpy_fn_ptrs, bool use_cuda_graph) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph
    cuda_ctx->cuda_graph->updated_kernel_arg.clear();
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
            use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
        }

        if (node->op == GGML_OP_MUL_MAT_ID) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to mul_mat_id\n", __func__);
#endif
        }

        if (node->op == GGML_OP_ADD && node->src[1] && node->src[1]->ne[1] > 1) {
            // disable CUDA graphs for batch size > 1 for now.
            // Changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (node->op == GGML_OP_CPY) {
            // store the copy op parameter which changes with each token.
            cuda_ctx->cuda_graph->updated_kernel_arg.push_back((char **) &(node->src[1]->data));
            // store a pointer to each copy op CUDA kernel to identify it later
            void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
            if (!ptr) {
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
            } else {
                if (std::find(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), ptr) == ggml_cuda_cpy_fn_ptrs.end()) {
                    ggml_cuda_cpy_fn_ptrs.push_back(ptr);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address &&
          node->op != GGML_OP_CPY &&
          node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != graph_node_properties->src_address[i] &&
            node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW
        ) {
            return false;
        }
    }

    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static void maintain_cuda_graph(ggml_backend_cuda_context * cuda_ctx, std::vector<void *> & ggml_cuda_cpy_fn_ptrs, bool cuda_graph_update_required) {

    if (cuda_graph_update_required) {
        // Extract nodes from graph
        // First call with null argument gets number of nodes in graph
        CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, nullptr, &cuda_ctx->cuda_graph->num_nodes));
        // Subsequent call with non-null argument gets nodes
        cuda_ctx->cuda_graph->nodes.clear();
        cuda_ctx->cuda_graph->nodes.resize(cuda_ctx->cuda_graph->num_nodes);
        cuda_ctx->cuda_graph->params.clear();
        cuda_ctx->cuda_graph->params.resize(cuda_ctx->cuda_graph->num_nodes);
        if (cuda_ctx->cuda_graph->num_nodes > 0) {
            CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, cuda_ctx->cuda_graph->nodes.data(), &cuda_ctx->cuda_graph->num_nodes));

            // Loop over nodes, and extract kernel parameters from each node
            for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                cudaGraphNodeType node_type;
                CUDA_CHECK(cudaGraphNodeGetType(cuda_ctx->cuda_graph->nodes[i], &node_type));
                if (node_type == cudaGraphNodeTypeKernel) {
                    cudaError_t stat = cudaGraphKernelNodeGetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]); // Get params using runtime
                    if (stat == cudaErrorInvalidDeviceFunction) {
                        // Fails due to incorrect handling by CUDA runtime of CUDA BLAS node.
                        // We don't need to update blas nodes, so clear error and move on.
                        (void)cudaGetLastError();
                    } else {
                        GGML_ASSERT(stat == cudaSuccess);
                    }
                }
            }
        }
    } else {
        // One of the arguments to the copy kernel is updated for each token, hence we need to
        // replace that argument with the updated value in the CUDA graph
        // on update steps, the live parameters will already be captured
        int k = 0;
        for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
            if(count(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), cuda_ctx->cuda_graph->params[i].func) > 0) {
                char ** updated_kernel_arg_ptr = cuda_ctx->cuda_graph->updated_kernel_arg.at(k++);
                cuda_ctx->cuda_graph->params[i].kernelParams[1] = updated_kernel_arg_ptr;
                CUDA_CHECK(cudaGraphKernelNodeSetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]));
            }
        }
    }
}

static bool is_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {

    bool cuda_graph_update_required = false;

    if (cuda_ctx->cuda_graph->instance == nullptr) {
        cuda_graph_update_required = true;
    }

    // Check if the graph size has changed
    if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        cuda_graph_update_required = true;
        cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool has_matching_properties = true;
        if (!cuda_graph_update_required) {
            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            cuda_graph_update_required = true;
        }
        set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
    }

    return cuda_graph_update_required;
}

static void update_cuda_graph_executable(ggml_backend_cuda_context * cuda_ctx) {

    cudaGraphExecUpdateResultInfo result_info;
#ifdef __HIP_PLATFORM_AMD__
    hipGraphNode_t errorNode;
    hipError_t stat = hipGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &errorNode, &result_info);
#else
    cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &result_info);
#endif
    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(cuda_ctx->cuda_graph->instance));
        cuda_ctx->cuda_graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif

static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
   [[maybe_unused]] std::vector<void *> & ggml_cuda_cpy_fn_ptrs,  bool & graph_evaluated_or_captured, bool & use_cuda_graph,
    bool & cuda_graph_update_required) {

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft));
                    }
                }
#endif

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        }

        // Perform update to graph (if required for this token), and change copy parameter (required for every token)
        maintain_cuda_graph(cuda_ctx, ggml_cuda_cpy_fn_ptrs, cuda_graph_update_required);

        // Update graph executable
        update_cuda_graph_executable(cuda_ctx);

        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    // vector of pointers to CUDA cpy kernels, which are required to identify
    // kernel parameters which need updated in the graph for each token
    std::vector<void *> ggml_cuda_cpy_fn_ptrs;

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;

    if (cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(cuda_ctx, cgraph);

        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, cgraph,
                             ggml_cuda_cpy_fn_ptrs, use_cuda_graph);

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) { // Start CUDA graph capture
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, ggml_cuda_cpy_fn_ptrs, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemGetInfo(free, total));
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                // for small weight matrices the active device can end up without any rows, don't use row split in those cases
                // this avoids some edge cases (and the performance would not be good anyways)
                if (a->buffer && ggml_backend_buft_is_cuda_split(a->buffer->buft)) {
                    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) a->buffer->buft->context;
                    int64_t row_low;
                    int64_t row_high;
                    get_row_split(&row_low, &row_high, a, buft_ctx->tensor_split, dev_ctx->device);
                    if (row_low == row_high) {
                        return false;
                    }
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                if (b->type == GGML_TYPE_F16 && b->ne[2]*b->ne[3] > 1 &&
                    !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    return false;
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
#ifdef GGML_USE_MUSA
                        if (a->type == GGML_TYPE_Q3_K) {
                            return false;
                        }
#endif // GGML_USE_MUSA
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16: // [jart]
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
            return true;
        case GGML_OP_CONT:
            return op->src[0]->type != GGML_TYPE_BF16;
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            const size_t ts = ggml_type_size(op->src[0]->type);
            const int64_t ne0_012 = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2];
            return op->src[0]->nb[0] == ts && op->src[0]->nb[3] == ne0_012*ts;
        }
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
            return true;
        case GGML_OP_FLASH_ATTN_EXT: {
#ifndef FLASH_ATTN_AVAILABLE
            return false;
#endif
#if defined(GGML_MINIMIZE_CODE_SIZE)
            return false;
#endif
            if (op->src[1]->type == GGML_TYPE_BF16 || op->src[2]->type == GGML_TYPE_BF16) {
                return false;
            }
            if (op->src[0]->ne[0] ==  64 && op->src[1]->type == GGML_TYPE_F16) {
                return true;
            }
            if (op->src[0]->ne[0] == 128) {
                return true;
            }
            if (op->src[0]->ne[0] == 256 && op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16) {
                return true;
            }
            return fp16_mma_available(ggml_cuda_info().devices[dev_ctx->device].cc) &&
                op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16;
        }
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return (ggml_backend_buft_is_cuda(buft) || ggml_backend_buft_is_cuda_split(buft)) && buft->device == dev;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return get_op_batch_size(op) >= min_batch_size;

    GGML_UNUSED(dev);
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_F16
        features.push_back({ "F16", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
        features.push_back({ "PEER_MAX_BATCH_SIZE", STRINGIFY(GGML_CUDA_PEER_MAX_BATCH_SIZE) });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_cuda_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;

            for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);

                ggml_cuda_set_device(i);
                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context   = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)

#ifndef GGML_MINIMIZE_CODE_SIZE

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP fattn-wmma-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// Old and deprecated WMMA FlashAttention implementation.
// It is still needed for Volta since the memory layout of NVIDIA tensor cores changed with Turing.
// Long-term the WMMA code should be replaced with a dedicated Volta implementation.


#ifdef FP16_MMA_AVAILABLE
#include <mma.h>
#endif // FP16_MMA_AVAILABLE

// D == head size, VKQ_stride == num VKQ rows calculated in parallel:
template<int D, int ncols, int nwarps, int VKQ_stride, int parallel_blocks, typename KQ_acc_t, bool use_logit_softcap>
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = ncols*(blockIdx.x / parallel_blocks); // Index of the first Q/QKV column to work on.
    const int ip  =        blockIdx.x % parallel_blocks;  // Index in group of blocks running for the same column in parallel.

    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::row_major> frag_a_K;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_a_V;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_b;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c_VKQ;

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t)/sizeof(half);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f   = (const float *) (Q + nb02* blockIdx.y              + nb01*ic0);
    const half  * K_h   = (const half  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half  * V_h   = (const half  *) (V + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half  * maskh = (const half  *)  mask + (nb31/sizeof(half))* ic0;
    const half2 * mask2 = (const half2 *)  mask + (nb31/sizeof(half))*(ic0/2);

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);
    const half2 slope2 = make_half2(slopef, slopef);

    const half2 logit_softcap_2 = make_half2(logit_softcap, logit_softcap);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols*kqs_padded*kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio*ncols*D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    float * KQ_f = (float *) KQ;
    half2 * KQ2 = (half2 *) KQ;

    float    KQ_rowsum_f[ncols/nwarps] = {0.0f};
    float       KQ_max_f[ncols/nwarps];
    float KQ_max_scale_f[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX/2.0f;
    }

    half2    KQ_rowsum_h2[ncols/nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols/nwarps];
    half2 KQ_max_scale_h2[ncols/nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-HALF_MAX_HALF, -HALF_MAX_HALF);
    }

    __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                break;
            }
            VKQ2[j*(D_padded/2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            KQ[j*D_padded + i] = ic0 + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
        }
    }

    __syncthreads();

    // Load Q into tensor core fragments/registers since it will be used frequently:
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            nvcuda::wmma::load_matrix_sync(Q_b[i0/16][j0/frag_n], KQ + j0*D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    for (int k_VKQ_0 = ip*FATTN_KQ_STRIDE; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(KQ_c[j], 0.0f);
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                nvcuda::wmma::load_matrix_sync(K_a, K_h + (k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync((KQ_acc_t *) KQ + j0*kqs_padded + i_KQ_0 + frag_m*threadIdx.y, KQ_c[j0/frag_n], kqs_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float KQ_f_tmp[FATTN_KQ_STRIDE / WARP_SIZE];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] = KQ_f[j*kqs_padded + k];

                    if (use_logit_softcap) {
                        KQ_f_tmp[k0/WARP_SIZE] = logit_softcap*tanhf(KQ_f_tmp[k0/WARP_SIZE]);
                    }
                }

                float KQ_max_new = KQ_max_f[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] += mask ? __half2float(slopeh*maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = warp_reduce_max(KQ_max_new);

                const float diff = KQ_max_f[j0/nwarps] - KQ_max_new;
                KQ_max_scale_f[j0/nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0/nwarps] = 0.0f;
                }
                KQ_max_f[j0/nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const float diff = KQ_f_tmp[k0/WARP_SIZE] - KQ_max_f[j0/nwarps];
                    KQ_f_tmp[k0/WARP_SIZE] = expf(diff);
                    if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ_f_tmp[k0/WARP_SIZE] = 0.0f;
                    }
                    KQ_rowsum_add += KQ_f_tmp[k0/WARP_SIZE];
                    KQ[j*(kqar*kqs_padded) + k] = KQ_f_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0/nwarps] = KQ_max_scale_f[j0/nwarps]*KQ_rowsum_f[j0/nwarps] + KQ_rowsum_add;
            } else {
                half2 KQ2_tmp[FATTN_KQ_STRIDE/(2*WARP_SIZE)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] = KQ2[j*(kqs_padded/2) + k];

                    if (use_logit_softcap) {
                        // There is no dedicated tangens hyperbolicus function for half2.
                        KQ2_tmp[k0/WARP_SIZE] = h2exp(KQ2_tmp[k0/WARP_SIZE]*make_half2(2.0f, 2.0f));
                        KQ2_tmp[k0/WARP_SIZE] = (KQ2_tmp[k0/WARP_SIZE] - make_half2(1.0f, 1.0f))
                                               /(KQ2_tmp[k0/WARP_SIZE] + make_half2(1.0f, 1.0f));

                        KQ2_tmp[k0/WARP_SIZE] *= logit_softcap_2;
                    }
                }

                half2 KQ_max_new = KQ_max_h2[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] += mask ? slope2*mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                    KQ_max_new = ggml_cuda_hmax2(KQ_max_new, KQ2_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = __half2half2(warp_reduce_max(ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0/nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0/nwarps] = h2exp(diff);
                const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                *((uint32_t *) &KQ_max_scale_h2[j0/nwarps]) &= ftz_mask;
                KQ_max_h2[j0/nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const half2 diff = KQ2_tmp[k0/WARP_SIZE] - KQ_max_h2[j0/nwarps];
                    KQ2_tmp[k0/WARP_SIZE] = h2exp(diff);
                    const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                    *((uint32_t *) &KQ2_tmp[k0/WARP_SIZE]) &= ftz_mask;
                    KQ_rowsum_add += KQ2_tmp[k0/WARP_SIZE];
                    KQ2[j*(kqs_padded/2) + k] = KQ2_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0/nwarps] = KQ_max_scale_h2[j0/nwarps]*KQ_rowsum_h2[j0/nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        frag_b KQ_b[FATTN_KQ_STRIDE/(VKQ_ratio*16)][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;
                nvcuda::wmma::load_matrix_sync(
                    KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                    KQ + j0*(kqar*kqs_padded) + k,
                    kqar*kqs_padded);
            }
        }

        frag_c_VKQ VKQ_c[D/VKQ_stride][ncols/frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(VKQ_c[i_VKQ_0/VKQ_stride][j], 0.0f);
            }

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;

                frag_a_V v_a;
                nvcuda::wmma::load_matrix_sync(v_a, V_h + (k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(VKQ_c[i_VKQ_0/VKQ_stride][j], v_a, KQ_b[k0/(VKQ_ratio*16)][j], VKQ_c[i_VKQ_0/VKQ_stride][j]);
                }
            }
        }

        __syncthreads();

        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols*D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync(
                    KQ + offset_k + j0*D_padded + i_KQ_0 + frag_m*(threadIdx.y/VKQ_ratio),
                    VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                    D_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            half2 VKQ_scale;
            if (std::is_same<KQ_acc_t, float>::value) {
                VKQ_scale = make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]);
            } else {
                VKQ_scale = KQ_max_scale_h2[j0/nwarps];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                    break;
                }

                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l) {
                    VKQ_add += KQ2[l*(ncols*D_padded/2) + j*(D_padded/2) + i];
                }
                VKQ2[j*(D_padded/2) + i] = VKQ_scale*VKQ2[j*(D_padded/2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= ne01) {
            return;
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0/nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0/nwarps]) + __high2float(KQ_rowsum_h2[j0/nwarps]);
        }

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            float dst_val = VKQ[j_VKQ*D_padded + i];
            if (parallel_blocks == 1) {
                dst_val /= KQ_rowsum_j;
            }
            dst[j_dst*gridDim.y*D + blockIdx.y*D + i] = dst_val;
        }

        if (parallel_blocks == 1 || threadIdx.x != 0) {
            continue;
        }

        float2 dst_meta_val;
        if (std::is_same<KQ_acc_t, float>::value) {
            dst_meta_val.x = KQ_max_f[j0/nwarps];
        } else {
            dst_meta_val.x = __low2float(KQ_max_h2[j0/nwarps]);
        }
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = dst_meta_val;
    }
#else
   NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
}

constexpr int get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2*get_max_power_of_2(x/2) : 1;
}

static_assert(get_max_power_of_2(1) == 1, "Test failed.");
static_assert(get_max_power_of_2(2) == 2, "Test failed.");
static_assert(get_max_power_of_2(4) == 4, "Test failed.");
static_assert(get_max_power_of_2(6) == 2, "Test failed.");

// Number of VKQ rows calculated in parallel:
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (get_max_power_of_2(D/frag_m) < nwarps ? get_max_power_of_2(D/frag_m) : nwarps)*frag_m;
}

static_assert(get_VKQ_stride(128, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride(128, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride(128, 4, 32) == 128, "Test failed.");
static_assert(get_VKQ_stride( 64, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride( 64, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 64, 4, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 80, 1, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 2, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 4, 16) ==  16, "Test failed.");

template <int D, int cols_per_block, typename KQ_acc_t>
void ggml_cuda_flash_attn_ext_wmma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    constexpr int nwarps = 4;

    constexpr int frag_m = cols_per_block == 8 && D % 32 == 0 ? 32 : 16;
    const int blocks_num_pb1 = ((Q->ne[1] + cols_per_block - 1) / cols_per_block)*Q->ne[2]*Q->ne[3];
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (4*blocks_num_pb1 < 2*nsm) {
        constexpr int parallel_blocks = 4;
        fattn_kernel_t fattn_kernel;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        } else {
            constexpr bool use_logit_softcap = true;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        }
        launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, 0, true, true);
        return;
    }
    if (2*blocks_num_pb1 < 2*nsm) {
        constexpr int parallel_blocks = 2;
        fattn_kernel_t fattn_kernel;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        } else {
            constexpr bool use_logit_softcap = true;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        }
        launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, 0, true, true);
        return;
    }
    constexpr int parallel_blocks = 1;
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<
            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<
            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
    }
    launch_fattn<D, cols_per_block, parallel_blocks, -1>(ctx, dst, fattn_kernel, nwarps, 0, true, true);
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    if (prec != GGML_PREC_DEFAULT) {
        if (Q->ne[1] <= 32 || Q->ne[0] > 128) {
            constexpr int cols_per_block = 16;
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                    break;
                case 256:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
                    break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                    break;
                // case 256:
                //     ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
                //     break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        }
        return;
    }

    if (Q->ne[1] <= 8 && Q->ne[0] % WARP_SIZE == 0) {
        constexpr int cols_per_block = 8;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, half>(ctx, dst);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 16;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, half>(ctx, dst);
                break;
            case 80:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, half>(ctx, dst);
                break;
            case 112:
                ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, half>(ctx, dst);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    constexpr int cols_per_block = 32;
    switch (Q->ne[0]) {
        case 64:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, half>(ctx, dst);
            break;
        case 80:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, half>(ctx, dst);
            break;
        case 96:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, half>(ctx, dst);
            break;
        case 112:
            ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, half>(ctx, dst);
            break;
        case 128:
            ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, half>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, half>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-f16-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-f16-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-f16-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-f16-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-f16-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-f16-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_0-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_0-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_0-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_0-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_1-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_1-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_1-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_1-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_1-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q4_1-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_0-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_0-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_0-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_0-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_0-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_0-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_1-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_1-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_1-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_1-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_1-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q5_1-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q8_0-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q8_0-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q8_0-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q8_0-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q8_0-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs128-q8_0-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs256-f16-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs64-f16-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs64-f16-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs64-f16-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs64-f16-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs64-f16-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f16-instance-hs64-f16-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-f16-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-f16-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-f16-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-f16-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-f16-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-f16-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_0-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_0-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_0-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_0-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_1-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_1-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_1-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_1-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_1-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q4_1-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_0-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_0-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_0-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_0-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_0-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_0-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_1-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_1-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_1-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_1-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_1-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q5_1-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q8_0-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q8_0-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q8_0-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q8_0-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q8_0-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs128-q8_0-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs256-f16-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs64-f16-f16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs64-f16-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs64-f16-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs64-f16-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs64-f16-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-vec-f32-instance-hs64-f16-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-mma-f16-instance-cpb16.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_MMA_F16_CASE(64, 16);
DECL_FATTN_MMA_F16_CASE(80, 16);
DECL_FATTN_MMA_F16_CASE(96, 16);
DECL_FATTN_MMA_F16_CASE(112, 16);
DECL_FATTN_MMA_F16_CASE(128, 16);
DECL_FATTN_MMA_F16_CASE(256, 16);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-mma-f16-instance-cpb32.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_MMA_F16_CASE(64, 32);
DECL_FATTN_MMA_F16_CASE(80, 32);
DECL_FATTN_MMA_F16_CASE(96, 32);
DECL_FATTN_MMA_F16_CASE(112, 32);
DECL_FATTN_MMA_F16_CASE(128, 32);
DECL_FATTN_MMA_F16_CASE(256, 32);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-mma-f16-instance-cpb64.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_MMA_F16_CASE(64, 64);
DECL_FATTN_MMA_F16_CASE(80, 64);
DECL_FATTN_MMA_F16_CASE(96, 64);
DECL_FATTN_MMA_F16_CASE(112, 64);
DECL_FATTN_MMA_F16_CASE(128, 64);
DECL_FATTN_MMA_F16_CASE(256, 64);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/fattn-mma-f16-instance-cpb8.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_FATTN_MMA_F16_CASE(64, 8);
DECL_FATTN_MMA_F16_CASE(80, 8);
DECL_FATTN_MMA_F16_CASE(96, 8);
DECL_FATTN_MMA_F16_CASE(112, 8);
DECL_FATTN_MMA_F16_CASE(128, 8);
DECL_FATTN_MMA_F16_CASE(256, 8);


#endif // GGML_MINIMIZE_CODE_SIZE

#ifndef GGML_NO_IQUANTS

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq1_s.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ1_S);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq2_s.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ2_S);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq2_xs.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq2_xxs.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq3_s.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ3_S);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq3_xxs.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq4_nl.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-iq4_xs.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);

#endif // GGML_NO_IQUANTS

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q2_k.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q2_K);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q3_k.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q3_K);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q4_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q4_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q4_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q4_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q4_k.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q4_K);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q5_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q5_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q5_1.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q5_1);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q5_k.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q5_K);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q6_k.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q6_K);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP template-instances/mmq-instance-q8_0.cu
//
////////////////////////////////////////////////////////////////////////////////

// This file has been autogenerated by generate_cu_files.py, do not edit manually.


DECL_MMQ_CASE(GGML_TYPE_Q8_0);

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP cross-entropy-loss.cu
//
////////////////////////////////////////////////////////////////////////////////



template <bool use_shared>
static __global__ void cross_entropy_loss_f32(
        const float * __restrict__ logits, const float * __restrict__ labels, float * __restrict__ dst, const int nclasses, const int k) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x)*nclasses;
    labels += int64_t(blockIdx.x)*nclasses;

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        max_logit = fmaxf(max_logit, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    max_logit = warp_reduce_max(max_logit);

    // Calculate log(softmax(logits)) which is just logits - max:
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        sum += expf(logit_i - max_logit);
    }
    sum = warp_reduce_sum(sum);
    sum = logf(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        loss += (logit_i - max_logit - sum) * labels[i];
    }
    loss = -warp_reduce_sum(loss) / (float)k;

    if (threadIdx.x != 0) {
        return;
    }

    dst[blockIdx.x] = loss;
}

template <bool use_shared>
static __global__ void cross_entropy_loss_back_f32(
        const float * __restrict__ grad, const float * __restrict__ logits, const float * __restrict__ labels,
        float * __restrict__ dst, const int nclasses) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x)*nclasses;
    labels += int64_t(blockIdx.x)*nclasses;
    dst    += int64_t(blockIdx.x)*nclasses;

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        maxval = fmaxf(maxval, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf((use_shared ? tmp[i] : logits[i]) - maxval);
        sum += val;

        if (use_shared) {
            tmp[i] = val;
        } else {
            dst[i] = val;
        }
    }
    sum = warp_reduce_sum(sum);
    const float sm_scale = 1.0f/sum;

    const float d_by_nrows = *grad/gridDim.x;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = use_shared ? tmp[i] : dst[i];
        dst[i] = (val*sm_scale - labels[i])*d_by_nrows;
    }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    if (nbytes_shared <= smpbo) {
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(cross_entropy_loss_back_f32<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
            shared_memory_limit_raised[id] = true;
        }
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
        cross_entropy_loss_f32<true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);
    } else {
        cross_entropy_loss_f32<false><<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];

    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    const float * src1f_d = (const float *) src1f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo) {
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(cross_entropy_loss_back_f32<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
            shared_memory_limit_raised[id] = true;
        }
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
        cross_entropy_loss_back_f32<true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(grad_d, src0f_d, src1f_d, dst_d, ne00);
    } else {
        cross_entropy_loss_back_f32<false><<<blocks_num, blocks_dim, 0, stream>>>(grad_d, src0f_d, src1f_d, dst_d, ne00);
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP sum.cu
//
////////////////////////////////////////////////////////////////////////////////

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB



void sum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream) {
#ifdef USE_CUB
    size_t tmp_size = 0;
    DeviceReduce::Sum(nullptr,       tmp_size, x, dst, ne, stream);
    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
    DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);
#else
    // Use (inefficient) sum_rows implementation as a fallback.
    // For AMD there is rocPRIM which could be used as a drop-in replacement via hipcub but this would require C++11 -> C++14.
    sum_rows_f32_cuda(x, dst, ne, 1, stream);
    GGML_UNUSED(pool);
#endif // USE_CUB
}

void ggml_cuda_op_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne = ggml_nelements(src0);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    sum_f32_cuda(pool, src0_d, dst_d, ne, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP opt-step-adamw.cu
//
////////////////////////////////////////////////////////////////////////////////

#include "ggml-impl.h"


static __global__ void opt_step_adamw_f32(
    float * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m, float * __restrict__ g_v,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[i];
    const float gmi = g_m[i]*beta1 +    gi*(1.0f - beta1);
    const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    const float mh =       gmi*beta1h;
    const float vh = sqrtf(gvi*beta2h) + eps;

    x[i] = x[i]*(1.0f - alpha*wd) - alpha*mh/vh;
}

static void opt_step_adamw_f32_cuda(
    float * x, const float * g, float * g_m, float * g_v, const float * pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    opt_step_adamw_f32<<<block_nums, block_dims, 0, stream>>>(x, g, g_m, g_v, pars, k);
}

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0->type         == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type    == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_m->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_v->type  == GGML_TYPE_F32);
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
    GGML_ASSERT(ggml_is_contiguous(adamw_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    float       * src0_d         = (float       *) src0->data;
    const float * src0_grad_d    = (const float *) src0_grad->data;
    float       * src0_grad_m_d  = (float       *) src0_grad_m->data;
    float       * src0_grad_v_d  = (float       *) src0_grad_v->data;
    const float * adamw_params_d = (const float *) adamw_params->data;

    cudaStream_t stream = ctx.stream();

    const int64_t ne = ggml_nelements(src0);

    opt_step_adamw_f32_cuda(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP out-prod.cu
//
////////////////////////////////////////////////////////////////////////////////



void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t lda = nb01 / sizeof(float);
    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));

    // data strides in dimensions 2/3
    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    // TODO batched matrix multiplication
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                src1_d +  i3      *s13 +  i2      *s12, ldb,
                        &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP wkv6.cu
//
////////////////////////////////////////////////////////////////////////////////


static __global__ void rwkv_wkv_f32(const int B, const int T, const int C, const int H, const float * k, const float * v, const float * r, const float * tf, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = CUDA_WKV_BLOCK_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _tf[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    __syncthreads();
    _tf[tid] = tf[head_i * head_size + tid];
    __syncthreads();

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& tf = (float4&)(_tf[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            y += r.x * (tf.x * kv.x + s.x);
            y += r.y * (tf.y * kv.y + s.y);
            y += r.z * (tf.z * kv.z + s.z);
            y += r.w * (tf.w * kv.w + s.w);

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * tf_d = (const float *)dst->src[3]->data;
    const float * td_d = (const float *)dst->src[4]->data;
    const float * s_d  = (const float *)dst->src[5]->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE); // The current cuda kernel is designed for RWKV6, HEAD_SIZE == 64

    rwkv_wkv_f32<<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP argmax.cu
//
////////////////////////////////////////////////////////////////////////////////



static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP count-equal.cu
//
////////////////////////////////////////////////////////////////////////////////


template <typename T>
static __global__ void count_equal(const T * __restrict__ x, const T * __restrict__ y, int64_t * __restrict__ dst, const int64_t dk, const int64_t k) {
    const int64_t i0 = (int64_t) blockIdx.x*dk;
    const int64_t i1 = min(i0 + dk, k);

    int nequal = 0;

    for (int64_t i = i0 + threadIdx.x; i < i1; i += WARP_SIZE) {
        const T xi = x[i];
        const T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum(nequal);

    if (threadIdx.x != 0) {
        return;
    }

    atomicAdd((int *) dst, nequal);
}

void ggml_cuda_count_equal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    const int64_t dne = GGML_PAD((ne + 4*nsm - 1) / (4*nsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);

    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(std::min((int64_t)4*nsm, (ne + CUDA_COUNT_EQUAL_CHUNK_SIZE - 1)/CUDA_COUNT_EQUAL_CHUNK_SIZE), 1, 1);

    switch (src0->type) {
        case GGML_TYPE_I32: {
            const int * src0_d = (const int *) src0->data;
            const int * src1_d = (const int *) src1->data;
            count_equal<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, dst_d, dne, ne);
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ROLLUP gla.cu
//
////////////////////////////////////////////////////////////////////////////////


template<int HEAD_SIZE>
static __global__ void gated_linear_attn_f32(const int B, const int T, const int C, const int H, const float scale,
     const float * k, const float * v, const float * r, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = HEAD_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4 & k = (float4 &)(_k[j]);
            const float4 & r = (float4 &)(_r[j]);
            const float4 & td = (float4 &)(_td[j]);
            float4 & s = (float4 &)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;

            y += r.x * s.x;
            y += r.y * s.y;
            y += r.z * s.z;
            y += r.w * s.w;
        }
        dst[t] = y * scale;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

void ggml_cuda_op_gated_linear_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * td_d = (const float *)dst->src[3]->data;
    const float * s_d  = (const float *)dst->src[4]->data;

    const int64_t B = dst->src[4]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float scale;
    memcpy(&scale, (float*)dst->op_params, sizeof(float));

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);


    if (C / H == 64) {
        gated_linear_attn_f32<64><<<B * H, C / H, 0, stream>>>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } else {
        gated_linear_attn_f32<128><<<B * H, C / H, 0, stream>>>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
