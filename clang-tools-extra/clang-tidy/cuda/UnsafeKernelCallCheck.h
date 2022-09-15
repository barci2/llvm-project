//===--- UnsafeKernelCallCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUDA_UNSAFEKERNELCALLCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUDA_UNSAFEKERNELCALLCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringSet.h"
#include <unordered_set>

namespace clang {
namespace tidy {
namespace cuda {

/// Checks for whether the possible errors with kernel launches are handled.
///
/// CUDA kernels do not always launch correctly. This may happen due to a driver
/// malfunction, lack of permissions, lack of a GPU, or a multitude of other
/// reasons. Such errors should be detected by calling the cudaGetLastError()
/// function following the kernel invocation. The invocation of the error should
/// be the the first side-effectful AST node after the invocation of the kernel
/// call (traversing the AST post-order) and a part of the first non-expression
/// statement after the kernel call. More precisely, it should be the first CFG
/// statement produced in line after the kernel call using the default options
/// for CFG building. This is because having the error checks closer to the
/// kernel invocation makes it easier to debug the code.
///
/// The check provides the following options:
///  - "HandlerName" (optional):
///      specifies the name of the function or the macro to which the return
///      value of the API call should be passed. This effectively automates the
///      process of adding the error checks in question for projects that have
///      such a mechanism implemented in them. The handler will also be accepted
///      even if it does not actually call cudaGetLastError().
///  - "AcceptedHandlers" (optional):
///      a comma-separated list specifying the only accepted handling
///      functions/macros that can alternatively handle the kernel error besides
///      the handler specified in HandlerName. The handlers may have scope
///      specifiers included in them, but if so then the full qualified name
///      (with all namespaces explicitly stated) has to be provided (for the
///      performance sake).
class UnsafeKernelCallCheck : public ClangTidyCheck {
  class PPCallback;

public:
  UnsafeKernelCallCheck(llvm::StringRef Name,
                        clang::tidy::ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(clang::ast_matchers::MatchFinder *Finder) override;
  void
  check(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::string HandlerName;
  void reportIssue(const Stmt &Stmt, ASTContext &Context);
  bool checkHandlerMacro(const Stmt &Stmt, ASTContext &Context);

  const std::string AcceptedHandlersList;
  const llvm::StringSet<llvm::MallocAllocator> AcceptedHandlersSet;
  bool isAcceptedHandler(const StringRef &Name);
  static llvm::StringSet<llvm::MallocAllocator>
  splitAcceptedHandlers(const llvm::StringRef &AcceptedHandlers,
                        const llvm::StringRef &HandlerName);

  std::unordered_set<SourceLocation,
                     std::function<unsigned(const SourceLocation &)>>
      HandlerMacroLocations;
};

} // namespace cuda
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUDA_UNSAFEKERNELCALLCHECK_H
