#include "clang-tools-extra/test/clang-tidy/checkers/Inputs/Headers/cuda/cuda_runtime.h"

__global__
void kernel();

void foo() {
  kernel<<<64, 128>>>();
}
