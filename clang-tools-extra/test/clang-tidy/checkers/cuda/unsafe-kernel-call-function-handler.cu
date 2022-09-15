// RUN: %check_clang_tidy %s cuda-unsafe-kernel-call %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: cuda-unsafe-kernel-call.HandlerName, \
// RUN:               value: 'errorCheck'}] \
// RUN:             }" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cuda/cuda_runtime.h>

__global__
void b();

void general();

void errorCheck() {
  auto err = cudaGetLastError();
}

void bad_next_line_stmt() {
  b<<<1, 2>>>();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  // CHECK-FIXES: {{^}}  errorCheck();{{$}}
  general();

  b<<<1, 2>>>(); /* some */ /* comments */ // present
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  // CHECK-FIXES: {{^}}  errorCheck();{{$}}
  general();

  if (true) // Dummy comment
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}  if (true) { // Dummy comment{{$}}
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>();{{$}}
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
  else // Dummy comment
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}  } else { // Dummy comment{{$}}
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>();{{$}}
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    // CHECK-FIXES: {{^}}  }{{$}}
  general();

  while (true) b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}  while (true) { b<<<1, 2>>>();{{$}}
    // CHECK-FIXES: {{^}}                 errorCheck();{{$}}
    // CHECK-FIXES: {{^}}  }{{$}}
  general();

  for (;;) // Dummy comment
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}  for (;;) { // Dummy comment{{$}}
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>();{{$}}
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    // CHECK-FIXES: {{^}}  }{{$}}
  general();

  if (true) {
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    general();
  } else {
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    general();
  }

  while(true) {
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    general();
  }

  for (;;) {
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    general();
  }

  do {
    b<<<1, 2>>>();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    errorCheck();{{$}}
    general();
  } while(true);
}

void bad_same_line_stmt() {
  b<<<1, 2>>>(); general();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  // CHECK-FIXES: {{^}}  b<<<1, 2>>>(); errorCheck(); general();{{$}}

  b<<<1, 2>>>(); /* hello */ /* there */ general(); // kenobi
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
  // CHECK-FIXES: {{^}}  b<<<1, 2>>>(); errorCheck(); /* hello */ /* there */ general(); // kenobi{{$}}

  if (true) // Dummy comment
    b<<<1, 2>>>(); /* comment */ general(); // comment
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    {b<<<1, 2>>>(); errorCheck();} /* comment */ general(); // comment{{$}}

  while (true) b<<<1, 2>>>(); general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}  while (true) {b<<<1, 2>>>(); errorCheck();} general();{{$}}

  for (;;) // Dummy comment
    b<<<1, 2>>>(); /* comment */ general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    {b<<<1, 2>>>(); errorCheck();} /* comment */ general();{{$}}

  if (true) {
    b<<<1, 2>>>(); general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>(); errorCheck(); general();{{$}}
  } else {
    b<<<1, 2>>>(); /* comment */ general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>(); errorCheck(); /* comment */ general();{{$}}
  }

  while(true) {
    b<<<1, 2>>>(); /* comment */ general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>(); errorCheck(); /* comment */ general();{{$}}
  }

  for (;;) {
    b<<<1, 2>>>(); general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>(); errorCheck(); general();{{$}}
  }

  do {
    b<<<1, 2>>>(); /* comment */ general();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Possible unchecked error after a kernel launch.
    // CHECK-FIXES: {{^}}    b<<<1, 2>>>(); errorCheck(); /* comment */ general();{{$}}
  } while(true);
}

void good() {
  b<<<1, 2>>>();
  errorCheck(); // Here the function call works because the handler is set to its name
}
