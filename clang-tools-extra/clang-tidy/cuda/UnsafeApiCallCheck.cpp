//===--- UnsafeApiCallCheck.cpp - clang-tidy-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeApiCallCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/FixIt.h"

#include <functional>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cuda {

namespace {

constexpr auto HandlerNameOptionName = "HandlerName";
constexpr auto AcceptedHandlersOptionName = "AcceptedHandlers";

} // namespace

UnsafeApiCallCheck::UnsafeApiCallCheck(llvm::StringRef Name,
                                       clang::tidy::ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      HandlerName(Options.get(HandlerNameOptionName, "")),
      AcceptedHandlersList(Options.get(AcceptedHandlersOptionName, "")),
      AcceptedHandlersSet(
          splitAcceptedHandlers(AcceptedHandlersList, HandlerName)),
      AcceptedHandlerMacroLocations(
          8, [](const SourceLocation &sLoc) { return sLoc.getHashValue(); }) {
  // If an empty string was inserted then that means that there was an empty
  // accepted handler in the list
  if (AcceptedHandlersSet.find("") != AcceptedHandlersSet.end()) {
    configurationDiag(
        "Empty handler name found in the list of accepted handlers",
        DiagnosticIDs::Error);
  }
}

llvm::StringSet<llvm::MallocAllocator>
UnsafeApiCallCheck::splitAcceptedHandlers(
    const llvm::StringRef &AcceptedHandlers,
    const llvm::StringRef &HandlerName) {
  // Check the case for when the accepted handlers are empty since otherwise
  // split(...) will still fill the vector with an empty element
  if (AcceptedHandlers.trim().empty()) {
    return llvm::StringSet<llvm::MallocAllocator>();
  }
  llvm::SmallVector<llvm::StringRef> AcceptedHandlersVector;
  AcceptedHandlers.split(AcceptedHandlersVector, ',');

  llvm::StringSet<llvm::MallocAllocator> AcceptedHandlersSet;
  for (auto AcceptedHandler : AcceptedHandlersVector) {
    AcceptedHandlersSet.insert(AcceptedHandler.trim());
  }

  // If the handler for FixItHints is set then add it to
  if (!AcceptedHandlersSet.empty() && !HandlerName.empty()) {
    AcceptedHandlersSet.insert(HandlerName);
  }

  return AcceptedHandlersSet;
}

void UnsafeApiCallCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, HandlerNameOptionName, HandlerName);
  Options.store(Opts, AcceptedHandlersOptionName, AcceptedHandlersList);
}

inline bool UnsafeApiCallCheck::limitAcceptedHandlers() {
  return !AcceptedHandlersSet.empty();
}

// Used for finding the occurences of accepted handler macros in the source
// code.
class UnsafeApiCallCheck::PPCallbacks : public clang::PPCallbacks {
public:
  PPCallbacks(UnsafeApiCallCheck *Check) : Check(Check) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    if (Check->AcceptedHandlersSet.find(
            MacroNameTok.getIdentifierInfo()->getName()) !=
        Check->AcceptedHandlersSet.end()) {
      Check->AcceptedHandlerMacroLocations.insert(MacroNameTok.getLocation());
    }
  }

private:
  UnsafeApiCallCheck *Check;
};

void UnsafeApiCallCheck::registerPPCallbacks(const SourceManager &SM,
                                             Preprocessor *PP,
                                             Preprocessor *ModuleExpanderPP) {
  if (limitAcceptedHandlers()) {
    ModuleExpanderPP->addPPCallbacks(std::make_unique<PPCallbacks>(this));
  }
}

namespace {

// Check if the declaration is in a specific header based on a condition
AST_MATCHER_P(Decl, isInSourceFile, std::function<bool(const StringRef &)>,
              SourceFileNameCond) {
  auto Loc = Node.getLocation();
  const auto &SM = Finder->getASTContext().getSourceManager();
  while (Loc.isValid()) {
    if (SourceFileNameCond(SM.getFilename(Loc))) {
      return true;
    }
    Loc = SM.getIncludeLoc(SM.getFileID(Loc));
  }
  return false;
}

// Check if the name of the declaration matches a specific condition
AST_MATCHER_P(NamedDecl, hasName, std::function<bool(const StringRef &)>,
              DeclNameCond) {
  return DeclNameCond(Node.getName());
}

// Check if the fully qualified name of the declaration matches a specific
// condition
AST_MATCHER_P(NamedDecl, hasQualName, std::function<bool(const StringRef &)>,
              DeclNameCond) {
  return DeclNameCond(Node.getQualifiedNameAsString());
}

constexpr auto UnusedValueBinding = "UnusedValueCall";
constexpr auto badlyHandledBinding = "badlyHandledCall";

// Common matchers for both unlimited and limited accepted handlers.
const auto HostFunction = functionDecl(unless(anyOf(
    hasAttr(attr::CUDADevice),
    hasAttr(attr::CUDAGlobal)))); // CUDA API cannot be called from device code
const auto ApiCallExpression = callExpr(
    callee(functionDecl(isInSourceFile([](StringRef FileName) {
                          return FileName.endswith("cuda_runtime.h") ||
                                 FileName.endswith("cuda_runtime_wrapper.h");
                        }), // All CUDA API is included from the cuda_runtime.h
                            // header or __cuda_runtime_wrapper.h
                        returns(asString("cudaError_t")))));

} // namespace

void UnsafeApiCallCheck::UnusedValueCallback::run(
    const MatchFinder::MatchResult &Result) {
  auto Node = Result.Nodes.getNodeAs<Stmt>(UnusedValueBinding);
  assert(Node);
  Check->UnusedValueNodes.insert(Node);
}

void UnsafeApiCallCheck::UnusedValueCallback::onStartOfTranslationUnit() {
  Check->UnusedValueNodes.clear();
}

void UnsafeApiCallCheck::registerMatchers(MatchFinder *Finder) {
  if (limitAcceptedHandlers()) {
    registerBadlyHandledMatchers(Finder);
  } else {
    registerUnusedValueMatchers(Finder);
  }
}

void UnsafeApiCallCheck::registerUnusedValueMatchers(MatchFinder *Finder) {
  const auto UnusedValue =
      matchers::isValueUnused(stmt(ApiCallExpression.bind(UnusedValueBinding)));
  Finder->addMatcher(functionDecl(HostFunction, hasBody(UnusedValue)), this);
}

void UnsafeApiCallCheck::registerBadlyHandledMatchers(MatchFinder *Finder) {
  const auto UnusedValue =
      matchers::isValueUnused(stmt(ApiCallExpression.bind(UnusedValueBinding)));
  UnusedValueCallbackInstance = std::make_unique<UnusedValueCallback>(this);
  Finder->addMatcher(functionDecl(HostFunction, hasBody(UnusedValue)),
                     UnusedValueCallbackInstance.get());

  const auto AcceptedHandlerPred = [this](const StringRef &Name) {
    return AcceptedHandlersSet.contains(Name);
  };

  const auto AcceptedHandlerDecl = functionDecl(
      anyOf(hasName(AcceptedHandlerPred), hasQualName(AcceptedHandlerPred)));
  const auto AcceptedHandlerParent = callExpr(callee(AcceptedHandlerDecl));

  Finder->addMatcher(
      functionDecl(
          HostFunction,
          forEachDescendant(stmt(ApiCallExpression.bind(badlyHandledBinding),
                                 unless(hasParent(AcceptedHandlerParent))))),
      this);
}

namespace {

constexpr auto HandlerMsg = "Consider wrapping it with a call to "
                            "an error handler:";
constexpr auto NoHandlerMsg =
    "Consider adding logic to check if an error has "
    "been returned or specify the error handler for this project.";

inline bool isStmtInMacro(const Stmt *const Stmt) {
  return Stmt->getBeginLoc().isInvalid() || Stmt->getBeginLoc().isMacroID() ||
         Stmt->getEndLoc().isInvalid() || Stmt->getEndLoc().isMacroID();
}

} // namespace

void UnsafeApiCallCheck::check(const MatchFinder::MatchResult &Result) {
  if (limitAcceptedHandlers()) {
    checkBadHandler(Result);
  } else {
    checkUnusedValue(Result);
  }
}

void UnsafeApiCallCheck::checkUnusedValue(
    const MatchFinder::MatchResult &Result) {
  const std::string MessagePrefix = "Unchecked CUDA API call. ";

  const auto ApiCallNode = Result.Nodes.getNodeAs<Stmt>(UnusedValueBinding);
  assert(ApiCallNode);

  // This disables the check for arguments inside macros, since we assume that
  // such a macro is intended as a handler (even if it just passes the argument
  // right through)
  if (Result.SourceManager->isMacroArgExpansion(ApiCallNode->getBeginLoc())) {
    return;
  }

  if (HandlerName.empty()) {
    diag(ApiCallNode->getBeginLoc(), MessagePrefix + NoHandlerMsg);
  } else if (isStmtInMacro(ApiCallNode)) {
    diag(ApiCallNode->getBeginLoc(),
         MessagePrefix + "Consider wrapping it with a call to `" + HandlerName +
             "`");
  } else {
    diag(ApiCallNode->getBeginLoc(), MessagePrefix + HandlerMsg)
        << FixItHint::CreateReplacement(
               ApiCallNode->getSourceRange(),
               (HandlerName + "(" +
                tooling::fixit::getText(ApiCallNode->getSourceRange(),
                                        *Result.Context) +
                ")")
                   .str());
  }
}

void UnsafeApiCallCheck::checkBadHandler(
    const MatchFinder::MatchResult &Result) {
  const std::string MessagePrefix = "CUDA API call not checked properly. ";

  const auto ApiCallNode = Result.Nodes.getNodeAs<Stmt>(badlyHandledBinding);
  assert(ApiCallNode);

  const auto ApiCallNodeMacroLocation = Result.SourceManager->getExpansionLoc(
      Result.SourceManager->getMacroArgExpandedLocation(
          ApiCallNode->getBeginLoc()));

  // This disables the check for arguments inside macros, since we assume that
  // such a macro is intended as a handler (even if it just passes the argument
  // right through)
  if (Result.SourceManager->isMacroArgExpansion(ApiCallNode->getBeginLoc()) &&
      AcceptedHandlerMacroLocations.find(ApiCallNodeMacroLocation) !=
          AcceptedHandlerMacroLocations.end()) {
    return;
  }

  if (HandlerName.empty()) {
    diag(ApiCallNode->getBeginLoc(), MessagePrefix + NoHandlerMsg);
  } else if (isStmtInMacro(ApiCallNode) ||
             UnusedValueNodes.find(ApiCallNode) == UnusedValueNodes.end()) {
    diag(ApiCallNode->getBeginLoc(),
         MessagePrefix + "Consider wrapping it with a call to `" + HandlerName +
             "`");
  } else {
    diag(ApiCallNode->getBeginLoc(), MessagePrefix + HandlerMsg)
        << FixItHint::CreateReplacement(
               ApiCallNode->getSourceRange(),
               (HandlerName + "(" +
                tooling::fixit::getText(ApiCallNode->getSourceRange(),
                                        *Result.Context) +
                ")")
                   .str());
  }
}

} // namespace cuda
} // namespace tidy
} // namespace clang
