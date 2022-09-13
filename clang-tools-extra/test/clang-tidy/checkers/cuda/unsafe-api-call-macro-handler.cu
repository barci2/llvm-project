// RUN: %check_clang_tidy %s cuda-unsafe-api-call %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: cuda-unsafe-api-call.HandlerName, \
// RUN:               value: 'CUDA_HANDLER'}] \
// RUN:             }" \
// RUN:   -- -isystem %clang_tidy_headers -std=c++14
#include <cuda/cuda_runtime.h>

class DummyContainer {
 public:
  int* begin();
  int* end();
};

#define DUMMY_CUDA_HANDLER(stmt) stmt
#define CUDA_HANDLER(stmt) do {auto err = stmt;} while(0)
#define API_CALL() do {cudaDeviceReset();} while(0)

void errorCheck();
void errorCheck(cudaError_t error);

void bad() {
  API_CALL();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
  // There isn't supposed to be a fix here since it's a macro call

  cudaDeviceReset();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
  // CHECK-FIXES:  {{^}}  CUDA_HANDLER(cudaDeviceReset());{{$}}
  errorCheck();

  if (true)
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset());{{$}}

  while (true)
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset());{{$}}

  do
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset());{{$}}
  while(false);

  switch (0) {
    case 0:
      cudaDeviceReset();
      // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
      // CHECK-FIXES:  {{^}}      CUDA_HANDLER(cudaDeviceReset());{{$}}
  }

  for(
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset());{{$}}
    ;
    cudaDeviceReset()
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset()){{$}}
  ) cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}  ) CUDA_HANDLER(cudaDeviceReset());{{$}}

  for(int i : DummyContainer())
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset());{{$}}

  auto x = ({
    cudaDeviceReset();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Unchecked CUDA API call.
    // CHECK-FIXES:  {{^}}    CUDA_HANDLER(cudaDeviceReset());{{$}}
    true;
  });
}

int good() {
  DUMMY_CUDA_HANDLER(cudaDeviceReset());

  if (cudaDeviceReset()) {
    return 0;
  }

  switch (cudaDeviceReset()) {
    case cudaErrorInvalidValue: return 1;
    case cudaErrorMemoryAllocation: return 2;
    default: return 3;
  }

  auto err = ({cudaDeviceReset();});
  // NOTE: We don't check that `errorCheck()` actually handles the error; we just assume it does.
  errorCheck(cudaDeviceReset());
}
