#ifndef GUFO_CORE_HIP_HIP_UTILS_HPP_
#define GUFO_CORE_HIP_HIP_UTILS_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <iostream>
#include <stdexcept>
#include <string>

#define HIP_CHECK(call)                                                 \
  do {                                                                  \
    hipError_t err = (call);                                            \
    if (err != hipSuccess) {                                            \
      std::string msg = std::string("HIP error in ") + __FILE__ + ":" + \
                        std::to_string(__LINE__) +                      \
                        " (" #call "): " + hipGetErrorString(err);      \
      std::cerr << msg << std::endl;                                    \
    }                                                                   \
  } while (0)

#define HIPBLAS_CHECK(call)                                                 \
  do {                                                                      \
    hipblasStatus_t status = (call);                                        \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
      std::string msg = std::string("hipBLAS error in ") + __FILE__ + ":" + \
                        std::to_string(__LINE__) +                          \
                        " (" #call "): " + std::to_string(status);          \
      std::cerr << msg << std::endl;                                        \
    }                                                                       \
  } while (0)

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_HIP_UTILS_HPP_
