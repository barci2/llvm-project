//===--- UnsafeApiCallCheck.h - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUDA_UNSAFEAPICALLCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUDA_UNSAFEAPICALLCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringSet.h"
#include <memory>
#include <unordered_set>

namespace clang {
namespace tidy {
namespace cuda {

/// Checks for whether the possible errors with the CUDA API invocations have
/// been handled.
///
/// Calls to CUDA API can sometimes fail to perform the action. This may happen
/// due to a driver malfunction, lack of permissions, lack of a GPU, or a
/// multitude of other reasons. Such errors are returned by those API calls and
/// should be handled in some way.
/// The check provides the following options:
///  - "HandlerName" (optional):
///      specifies the name of the function or the macro to which the return
///      value of the API call should be passed. This effectively automates the
///      process of adding the error checks in question for projects that have
///      such a mechanism implemented in them.
///  - "AcceptedHandlers" (optional):
///      a comma-separated list specifying the only accepted handling
///      functions/macros into which the error from the api call can be passed.
///      If not specified all ways to handle the error that do not just ignore
///      the output value are accepted. The handlers may have scope specifiers
///      included in them, but if so then the full qualified name (with all
///      namespaces explicitly stated) has to be provided (for the performance
///      sake). If the handler set in the "HandlerName" is not in the list of
///      accepted handlers then it gets added to it automatially.
///
/// Since the behavior of the check is significantly different when the
/// "AcceptedHandlers" option is set, the implementation is essentially split
/// into 2 paths, as highlighted by the comments near declarations.
class UnsafeApiCallCheck : public ClangTidyCheck {
  class PPCallbacks;

  // For gathering api calls with an unused value - only those nodes
  // can have a FixItHint when we limit the accepted handlers.
  //
  // Only used when "AcceptedHandlers" is set
  class UnusedValueCallback
      : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    UnusedValueCallback(UnsafeApiCallCheck *check) : Check(check) {}
    void
    run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
    void onStartOfTranslationUnit() override;

  private:
    UnsafeApiCallCheck *Check;
  };

public:
  UnsafeApiCallCheck(llvm::StringRef Name,
                     clang::tidy::ClangTidyContext *Context);

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(clang::ast_matchers::MatchFinder *Finder) override;
  void
  check(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::string HandlerName;

  // Only used when "AcceptedHandlers" is not set
  void registerUnusedValueMatchers(clang::ast_matchers::MatchFinder *Finder);
  // Only used when "AcceptedHandlers" is set
  void registerBadlyHandledMatchers(clang::ast_matchers::MatchFinder *Finder);

  // Only used when "AcceptedHandlers" is set
  void
  checkUnusedValue(const clang::ast_matchers::MatchFinder::MatchResult &Result);
  // Only used when "AcceptedHandlers" is not set
  void
  checkBadHandler(const clang::ast_matchers::MatchFinder::MatchResult &Result);

  const std::string AcceptedHandlersList; // Data store for AcceptedHandlersSet
  const llvm::StringSet<llvm::MallocAllocator> AcceptedHandlersSet;
  // Generates AcceptedHandlersSet from AcceptedHandlersList
  static llvm::StringSet<llvm::MallocAllocator>
  splitAcceptedHandlers(const llvm::StringRef &AcceptedHandlers,
                        const llvm::StringRef &HandlerName);
  bool limitAcceptedHandlers();

  // Only used when "AcceptedHandlers" is set
  std::unordered_set<SourceLocation,
                     std::function<unsigned(const SourceLocation &)>>
      AcceptedHandlerMacroLocations;
  std::unordered_set<const Stmt *> UnusedValueNodes;
  std::unique_ptr<UnusedValueCallback> UnusedValueCallbackInstance;
};

} // namespace cuda
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUDA_UNSAFEAPICALLCHECK_H
