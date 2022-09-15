// RUN: %check_clang_tidy %s cuda-unsafe-kernel-call %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: cuda-unsafe-kernel-call.HandlerName, \
// RUN:               value: 'CUDA_CHECK_KERNEL'}, \
// RUN:               {key: cuda-unsafe-kernel-call.AcceptedHandlers, \
// RUN:                value: 'ALTERNATIVE_CUDA_CHECK_KERNEL, cudaCheckKernel, \
// RUN:                        alternative::alternativeCudaCheckKernel, \
// RUN:                        otherAlternativeCudaCheckKernel'}] \
// RUN:             }" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cuda/cuda_runtime.h>

#define CUDA_CHECK_KERNEL() do {} while(0)

#define ALTERNATIVE_CUDA_CHECK_KERNEL() CUDA_CHECK_KERNEL()

void cudaCheckKernel();

namespace alternative {

void alternativeCudaCheckKernel();
void otherAlternativeCudaCheckKernel();

}

__global__
void b();

#define KERNEL_CALL() do {b<<<1, 2>>>();} while(0)

void errorCheck() {
  auto err = cudaGetLastError();
}

void bad() {
  b<<<1, 2>>>(); // sample comment
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.

  KERNEL_CALL(); // sample comment
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  // There isn't supposed to be a fix here since it's a macro call

  if(true)
    b<<<1, 2>>>()  ; // Brackets omitted purposefully, since they create an additional AST node
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  else {
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  }
  auto err = cudaGetLastError();

  b<<<1, 2>>>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  if (true)
    cudaGetLastError();

  b<<<1, 2>>>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  for(;;)
    auto err2 = cudaGetLastError(); // Brackets omitted purposefully, since they create an additional AST node

  b<<<1, 2>>>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  auto err3 = true ? 1 : cudaGetLastError();

  b<<<1, 2>>>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  auto err4 = cudaDeviceReset() + cudaGetLastError();

  b<<<1, 2>>>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  // Calling an error-checking function after a kernel is not considered safe.
  errorCheck();
}

void good() {
  b<<<1, 2>>>();; /* The semicolons are here because the
    detection of the macro is done with a lexer */ ;
  CUDA_CHECK_KERNEL();

  b<<<1, 2>>>();
  ALTERNATIVE_CUDA_CHECK_KERNEL();

  b<<<1, 2>>>();
  alternative::alternativeCudaCheckKernel();

  b<<<1, 2>>>();
  alternative::otherAlternativeCudaCheckKernel();

  b<<<1, 2>>>();
  switch(1 + cudaGetLastError()) {
    default:;
  }

  b<<<1, 2>>>();
  if(3 < cudaGetLastError()) {
    1;
  } else {
    2;
  }

  b<<<1, 2>>>();
  for(int i = cudaGetLastError();;);

  b<<<1, 2>>>();
  do {
  do {
  do {
    auto err2 = cudaGetLastError();
  } while(0);
  } while(0);
  } while(0);
}
