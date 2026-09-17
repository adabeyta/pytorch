#pragma once

// Header-only port of at::acc_type / at::acc_type_device from
// aten/src/ATen/AccumulateType.h. It lets custom kernels built against the
// stable ABI pick the type they should accumulate into without linking
// libtorch. The mapping tables below are copied from the ATen header and
// must stay in sync with it.
//
// Accumulating in a wider type than the input avoids precision loss (and
// overflow for small integer types) in reductions. The rules are:
//   * bool accumulates as bool.
//   * Floating types accumulate as float on CUDA (double is slow there),
//     except double itself, and as double on CPU. MPS never uses double.
//   * Integral types accumulate as int64_t.
//   * Complex types follow the same rules on their component type.
//
// Usage:
//   using acc_t = torch::headeronly::acc_type<c10::Half, /*is_cuda=*/true>;
//   using acc_t = torch::headeronly::acc_type_device<float,
//                     torch::headeronly::DeviceType::CPU>;

#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/complex.h>

#include <cstdint>

namespace torch::headeronly {

template <typename T, DeviceType D>
struct AccumulateTypeDevice {};

template <typename T, bool>
struct AccumulateType {};

template <typename T>
struct AccumulateType<T, false> {
  using type = typename AccumulateTypeDevice<T, DeviceType::CPU>::type;
};

template <typename T>
struct AccumulateType<T, true> {
  using type = typename AccumulateTypeDevice<T, DeviceType::CUDA>::type;
};

template <typename T, DeviceType device>
using acc_type_device = typename AccumulateTypeDevice<T, device>::type;

template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

// The macros are only used to build the tables below and are #undef'd at the
// end of this header so they do not clash with the unprefixed ATen ones.
#define THO_ACC_TYPE(t, acc_t, device_type)     \
  template <>                                   \
  struct AccumulateTypeDevice<t, device_type> { \
    using type = acc_t;                         \
  };
#define THO_MPS_ACC_TYPE(t, acc_t) THO_ACC_TYPE(t, acc_t, DeviceType::MPS)
#define THO_XPU_ACC_TYPE(t, acc_t) THO_ACC_TYPE(t, acc_t, DeviceType::XPU)
#define THO_CUDA_ACC_TYPE(t, acc_t) THO_ACC_TYPE(t, acc_t, DeviceType::CUDA)
#define THO_CPU_ACC_TYPE(t, acc_t) THO_ACC_TYPE(t, acc_t, DeviceType::CPU)

THO_MPS_ACC_TYPE(BFloat16, float)
THO_MPS_ACC_TYPE(Half, float)
THO_MPS_ACC_TYPE(Float8_e5m2, float)
THO_MPS_ACC_TYPE(Float8_e4m3fn, float)
THO_MPS_ACC_TYPE(Float8_e5m2fnuz, float)
THO_MPS_ACC_TYPE(Float8_e4m3fnuz, float)
THO_MPS_ACC_TYPE(float, float)
THO_MPS_ACC_TYPE(double, float)
THO_MPS_ACC_TYPE(int8_t, int64_t)
THO_MPS_ACC_TYPE(uint8_t, int64_t)
THO_MPS_ACC_TYPE(char, int64_t)
THO_MPS_ACC_TYPE(int16_t, int64_t)
THO_MPS_ACC_TYPE(int32_t, int64_t)
THO_MPS_ACC_TYPE(int64_t, int64_t)
THO_MPS_ACC_TYPE(bool, bool)
THO_MPS_ACC_TYPE(complex<Half>, complex<float>)
THO_MPS_ACC_TYPE(complex<float>, complex<float>)
THO_MPS_ACC_TYPE(complex<double>, complex<float>)

THO_XPU_ACC_TYPE(BFloat16, float)
THO_XPU_ACC_TYPE(Half, float)
THO_XPU_ACC_TYPE(Float8_e5m2, float)
THO_XPU_ACC_TYPE(Float8_e4m3fn, float)
THO_XPU_ACC_TYPE(Float8_e5m2fnuz, float)
THO_XPU_ACC_TYPE(Float8_e4m3fnuz, float)
THO_XPU_ACC_TYPE(float, float)
THO_XPU_ACC_TYPE(double, double)
THO_XPU_ACC_TYPE(int8_t, int64_t)
THO_XPU_ACC_TYPE(uint8_t, int64_t)
THO_XPU_ACC_TYPE(char, int64_t)
THO_XPU_ACC_TYPE(int16_t, int64_t)
THO_XPU_ACC_TYPE(int32_t, int64_t)
THO_XPU_ACC_TYPE(int64_t, int64_t)
THO_XPU_ACC_TYPE(bool, bool)
THO_XPU_ACC_TYPE(complex<Half>, complex<float>)
THO_XPU_ACC_TYPE(complex<float>, complex<float>)
THO_XPU_ACC_TYPE(complex<double>, complex<double>)

#if defined(__CUDACC__) || defined(__HIPCC__)
// The vendor fp16 type, available because Half.h pulls in cuda_fp16.h /
// hip_fp16.h under these compilers.
THO_CUDA_ACC_TYPE(half, float)
#endif
THO_CUDA_ACC_TYPE(BFloat16, float)
THO_CUDA_ACC_TYPE(Half, float)
THO_CUDA_ACC_TYPE(Float8_e5m2, float)
THO_CUDA_ACC_TYPE(Float8_e4m3fn, float)
THO_CUDA_ACC_TYPE(Float8_e5m2fnuz, float)
THO_CUDA_ACC_TYPE(Float8_e4m3fnuz, float)
THO_CUDA_ACC_TYPE(float, float)
THO_CUDA_ACC_TYPE(double, double)
THO_CUDA_ACC_TYPE(int8_t, int64_t)
THO_CUDA_ACC_TYPE(uint8_t, int64_t)
THO_CUDA_ACC_TYPE(char, int64_t)
THO_CUDA_ACC_TYPE(int16_t, int64_t)
THO_CUDA_ACC_TYPE(int32_t, int64_t)
THO_CUDA_ACC_TYPE(int64_t, int64_t)
THO_CUDA_ACC_TYPE(bool, bool)
THO_CUDA_ACC_TYPE(complex<Half>, complex<float>)
THO_CUDA_ACC_TYPE(complex<float>, complex<float>)
THO_CUDA_ACC_TYPE(complex<double>, complex<double>)

THO_CPU_ACC_TYPE(BFloat16, float)
THO_CPU_ACC_TYPE(Half, float)
THO_CPU_ACC_TYPE(Float8_e5m2, float)
THO_CPU_ACC_TYPE(Float8_e4m3fn, float)
THO_CPU_ACC_TYPE(Float8_e5m2fnuz, float)
THO_CPU_ACC_TYPE(Float8_e4m3fnuz, float)
THO_CPU_ACC_TYPE(float, double)
THO_CPU_ACC_TYPE(double, double)
THO_CPU_ACC_TYPE(int8_t, int64_t)
THO_CPU_ACC_TYPE(uint8_t, int64_t)
THO_CPU_ACC_TYPE(char, int64_t)
THO_CPU_ACC_TYPE(int16_t, int64_t)
THO_CPU_ACC_TYPE(int32_t, int64_t)
THO_CPU_ACC_TYPE(int64_t, int64_t)
THO_CPU_ACC_TYPE(bool, bool)
THO_CPU_ACC_TYPE(complex<Half>, complex<float>)
THO_CPU_ACC_TYPE(complex<float>, complex<double>)
THO_CPU_ACC_TYPE(complex<double>, complex<double>)

#undef THO_CPU_ACC_TYPE
#undef THO_CUDA_ACC_TYPE
#undef THO_XPU_ACC_TYPE
#undef THO_MPS_ACC_TYPE
#undef THO_ACC_TYPE

} // namespace torch::headeronly
