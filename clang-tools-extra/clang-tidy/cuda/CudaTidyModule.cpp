//===--- CudaTidyModule.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyCheck.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cuda {

class CudaModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {}
};

// Register the CudaTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<CudaModule>
    X("cuda-module", "Adds Cuda-related lint checks.");

} // namespace cuda

// This anchor is used to force the linker to link in the generated object file
// and thus register the CudaModule.
volatile int CudaModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
