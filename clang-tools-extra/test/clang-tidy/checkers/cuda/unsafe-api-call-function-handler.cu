// RUN: %check_clang_tidy %s cuda-unsafe-api-call %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: cuda-unsafe-api-call.HandlerName, \
// RUN:               value: 'cudaHandler'}, \
// RUN:              {key: cuda-unsafe-api-call.AcceptedHandlers, \
// RUN:               value: 'CUDA_HANDLER, DUMMY_CUDA_HANDLER, \
// RUN:                       alternative::cudaAlternativeHandler, \
// RUN:                       cudaOtherAlternativeHandler, bad::cudaBadHandler'}] \
// RUN:             }" \
// RUN:   -- -isystem %clang_tidy_headers -std=c++14
#include <cuda/cuda_runtime.h>

#define DUMMY_CUDA_HANDLER(stmt) stmt
#define CUDA_HANDLER(stmt) do {auto err = stmt;} while(0)
#define API_CALL() do {cudaDeviceReset();} while(0)
#define HANDLED_API_CALL() do {int err2 = cudaDeviceReset();} while(0)

void cudaHandler();
void cudaHandler(cudaError_t error);
void badCudaHandler(cudaError_t error);

namespace alternative {

void cudaAlternativeHandler(cudaError_t error);

void cudaOtherAlternativeHandler(cudaError_t error);

} // namespace alternative

void bad() {
  API_CALL();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
  // There isn't supposed to be a fix here since it's a macro call

  HANDLED_API_CALL();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
  // There isn't supposed to be a fix here since it's a macro call

  cudaDeviceReset();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
  // CHECK-FIXES:  {{^}}  cudaHandler(cudaDeviceReset());{{$}}
  cudaHandler();

  if (true)
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
    // CHECK-FIXES:  {{^}}    cudaHandler(cudaDeviceReset());{{$}}

  badCudaHandler(cudaDeviceReset());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
  // There isn't supposed to be a fix here since the result value is not unused

  int err = cudaDeviceReset();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
  // There isn't supposed to be a fix here since the result value is not unused

  if (cudaDeviceReset()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: CUDA API call not checked properly.
    // There isn't supposed to be a fix here since the result value is not unused
    return;
  }

}

void good() {
  cudaHandler(cudaDeviceReset());
  alternative::cudaAlternativeHandler(cudaDeviceReset());
  alternative::cudaOtherAlternativeHandler(cudaDeviceReset());
  CUDA_HANDLER(cudaDeviceReset() + 1);
  DUMMY_CUDA_HANDLER(cudaDeviceReset());
}
