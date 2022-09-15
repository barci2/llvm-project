//===--- UnsafeKernelCallCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeKernelCallCheck.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/FixIt.h"
#include <cctype>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cuda {

namespace {

constexpr auto HandlerNameOptionName = "HandlerName";
constexpr auto AcceptedHandlersOptionName = "AcceptedHandlers";

} // namespace

UnsafeKernelCallCheck::UnsafeKernelCallCheck(
    llvm::StringRef Name, clang::tidy::ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      HandlerName(Options.get(HandlerNameOptionName, "")),
      AcceptedHandlersList(Options.get(AcceptedHandlersOptionName, "")),
      AcceptedHandlersSet(
          splitAcceptedHandlers(AcceptedHandlersList, HandlerName)),
      HandlerMacroLocations(
          8, [](const SourceLocation &sLoc) { return sLoc.getHashValue(); }) {
  if (AcceptedHandlersSet.find("") != AcceptedHandlersSet.end()) {
    configurationDiag(
        "Empty handler name found in the list of accepted handlers",
        DiagnosticIDs::Error);
  }
}

llvm::StringSet<llvm::MallocAllocator>
UnsafeKernelCallCheck::splitAcceptedHandlers(
    const llvm::StringRef &AcceptedHandlers,
    const llvm::StringRef &HandlerName) {
  if (AcceptedHandlers.trim().empty()) {
    return HandlerName.empty()
               ? llvm::StringSet<llvm::MallocAllocator>()
               : llvm::StringSet<llvm::MallocAllocator>{HandlerName};
  }
  llvm::SmallVector<llvm::StringRef> AcceptedHandlersVector;
  AcceptedHandlers.split(AcceptedHandlersVector, ',');

  llvm::StringSet<llvm::MallocAllocator> AcceptedHandlersSet;
  for (auto AcceptedHandler : AcceptedHandlersVector) {
    AcceptedHandlersSet.insert(AcceptedHandler.trim());
  }
  if (!AcceptedHandlersSet.empty() && !HandlerName.empty()) {
    AcceptedHandlersSet.insert(HandlerName);
  }

  return AcceptedHandlersSet;
}

void UnsafeKernelCallCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, HandlerNameOptionName, HandlerName);
  Options.store(Opts, AcceptedHandlersOptionName, AcceptedHandlersList);
}

bool UnsafeKernelCallCheck::isAcceptedHandler(const StringRef &Name) {
  return AcceptedHandlersSet.contains(Name);
}

// Gathers the instances of the handler as a macro being used
class UnsafeKernelCallCheck::PPCallback : public PPCallbacks {
public:
  PPCallback(UnsafeKernelCallCheck &Check) : Check(Check) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    if (Check.isAcceptedHandler(MacroNameTok.getIdentifierInfo()->getName())) {
      Check.HandlerMacroLocations.insert(MacroNameTok.getLocation());
    }
  }

private:
  UnsafeKernelCallCheck &Check;
};

void UnsafeKernelCallCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  ModuleExpanderPP->addPPCallbacks(
      std::make_unique<UnsafeKernelCallCheck::PPCallback>(*this));
}

void UnsafeKernelCallCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(hasBody(hasDescendant(cudaKernelCallExpr())))
                         .bind("function"),
                     this);
}

namespace {

// Fetches the first parent available. Should be used
// for things that are common for the parents, like the location,
// since the only way a node can have multiple parents is with templates
template <typename Node, typename Parent = Node>
inline const Parent *getParent(const Node &Stmt, ASTContext &Context) {
  auto parents = Context.getParents(Stmt);

  return parents.empty() ? nullptr : parents.begin()->template get<Parent>();
}

bool isKernelCall(const Stmt *Stmt) {
  return Stmt->getStmtClass() == Stmt::CUDAKernelCallExprClass;
}

bool isInCudaRuntimeHeader(SourceLocation Loc, const SourceManager &SM) {
  constexpr auto CudaHeaderNameSuffix = "cuda_runtime.h";
  constexpr auto CudaWrapperHeaderNameSuffix = "cuda_runtime_wrapper.h";
  while (Loc.isValid()) {
    if (SM.getFilename(Loc).endswith(CudaHeaderNameSuffix) || SM.getFilename(Loc).endswith(CudaWrapperHeaderNameSuffix)) {
      return true;
    }
    Loc = SM.getIncludeLoc(SM.getFileID(Loc));
  }
  return false;
}

bool isCudaGetLastErrorCall(const Stmt *const Stmt, const SourceManager &SM) {
  constexpr auto GetLastErrorFunctionName = "cudaGetLastError";
  constexpr auto GetLastErrorFunctionScopedType = "::cudaError_t";
  constexpr auto GetLastErrorFunctionType = GetLastErrorFunctionScopedType + 2;
  if (Stmt->getStmtClass() != Stmt::CallExprClass) {
    return false;
  }
  auto CallExprNode = static_cast<const CallExpr *>(Stmt);

  if (!CallExprNode->getCalleeDecl() ||
      CallExprNode->getCalleeDecl()->getKind() != Decl::Function) {
    return false;
  }
  const auto FunctionDeclNode =
      static_cast<const FunctionDecl *>(CallExprNode->getCalleeDecl());

  const auto ReturnTypeName = FunctionDeclNode->getReturnType().getAsString();
  return FunctionDeclNode->getName() == GetLastErrorFunctionName &&
         (ReturnTypeName == GetLastErrorFunctionType ||
          StringRef(ReturnTypeName).endswith(GetLastErrorFunctionScopedType)) &&
         isInCudaRuntimeHeader(FunctionDeclNode->getLocation(), SM);
}

bool isHandlerCall(
    const Stmt *const Stmt,
    std::function<bool(const llvm::StringRef &)> HandlerNamePredicate) {
  if (Stmt->getStmtClass() != Stmt::CallExprClass) {
    return false;
  }
  auto CallExprNode = static_cast<const CallExpr *>(Stmt);

  if (!CallExprNode->getCalleeDecl() ||
      CallExprNode->getCalleeDecl()->getKind() != Decl::Function) {
    return false;
  }
  const auto FunctionDeclNode =
      static_cast<const FunctionDecl *>(CallExprNode->getCalleeDecl());

  return HandlerNamePredicate(FunctionDeclNode->getName()) ||
         HandlerNamePredicate(FunctionDeclNode->getQualifiedNameAsString());
}

/// Searches for the closest CFGElement that is an instance of CFGStmt. Does not
/// increment the index if it already indexes a CFGStmt.
const Stmt *findStmt(const CFGBlock *const Block, size_t &Idx) {
  while (Idx < Block->size() && !(*Block)[Idx].getAs<CFGStmt>().has_value()) {
    Idx++;
  }
  if (Idx < Block->size()) {
    return (*Block)[Idx].castAs<CFGStmt>().getStmt();
  }
  return nullptr;
}

inline bool isBlockReachable(const CFGBlock::AdjacentBlock &Block) {
  return Block && Block.isReachable();
}

template <typename Iter>
inline size_t countReachableBlocks(llvm::iterator_range<Iter> Range) {
  return std::count_if(Range.begin(), Range.end(), isBlockReachable);
}

template <typename Iter>
inline Iter findReachableBlock(llvm::iterator_range<Iter> Range) {
  return std::find_if(Range.begin(), Range.end(), isBlockReachable);
}

/// Searches for a next statement from this successor block as if all the empty
/// blocks were removed and all blocks that could be merged were merged. For
/// instance, in the following code the call to b() should be found assuming the
/// `block` argument is set to the first CFG block after the first block:
/// int foo() {
///   a();
///   do {
///     do {
///       b()
///     } while(0);
///   } while(0);
/// }
const Stmt *findNextStmtNonEmptyBlock(const CFGBlock *const Block) {
  // Enforce that the next block could be mergeable with the next block, i.e.
  // has no non-trivial predecesors. Trivial predecessors here are chains of
  // empty predecessors that have up to one predecessor that is itself a trivial
  // predecessor.
  int PrunedPredCount = 0;
  for (auto Pred : Block->preds()) {
    while (Pred && Pred.isReachable() && Pred->empty() &&
           countReachableBlocks(Pred->preds()) == 1) {
      Pred = *findReachableBlock(Pred->preds());
    }
    if (Pred && (!Pred->empty() || countReachableBlocks(Pred->preds()) > 1)) {
      ++PrunedPredCount;
    }
  }
  if (PrunedPredCount > 1) {
    return nullptr;
  }

  // Check if there is any statement in this block that we could return
  size_t Idx = 0;
  if (const auto Stmt = findStmt(Block, Idx)) {
    return Stmt;
  }

  // If the block is empty then try our luck with the next block, provided there
  // is only one
  if (countReachableBlocks(Block->succs()) != 1) {
    return nullptr;
  }
  const auto NextBlock = *findReachableBlock(Block->succs());
  return findNextStmtNonEmptyBlock(NextBlock);
}

} // namespace

void UnsafeKernelCallCheck::check(const MatchFinder::MatchResult &Result) {
  const auto FunctionDeclNode =
      Result.Nodes.getNodeAs<FunctionDecl>("function");
  const auto Cfg = CFG::buildCFG(FunctionDeclNode, FunctionDeclNode->getBody(),
                                 Result.Context, CFG::BuildOptions());

  for (const auto &block : *Cfg) {
    size_t Idx = 0;
    while (const auto Stmt = findStmt(block, Idx)) {
      ++Idx;
      if (!isKernelCall(Stmt)) {
        continue;
      }
      if (checkHandlerMacro(*Stmt, *Result.Context)) {
        continue;
      }

      auto NextStmt = findStmt(block, Idx);
      // Workaround for the do {...} while(0) not being erased out during
      // pruning
      if (!NextStmt) {
        if (countReachableBlocks(block->succs()) != 1) {
          reportIssue(*Stmt, *Result.Context);
          continue;
        }
        const auto NextBlock = findReachableBlock(block->succs());
        NextStmt = findNextStmtNonEmptyBlock(*NextBlock);
      }

      if (NextStmt && isCudaGetLastErrorCall(NextStmt, *Result.SourceManager)) {
        continue;
      }
      if (NextStmt &&
          isHandlerCall(NextStmt, [this](const llvm::StringRef &Name) {
            return isAcceptedHandler(Name);
          })) {
        continue;
      }
      reportIssue(*Stmt, *Result.Context);
    }
  }
}

// Searches for a handler macro being used right after the kernel call
bool UnsafeKernelCallCheck::checkHandlerMacro(const Stmt &Stmt,
                                              ASTContext &Context) {
  llvm::Optional<Token> Token = Lexer::findNextToken(
      Stmt.getEndLoc(), Context.getSourceManager(), Context.getLangOpts());
  if (!Token.has_value()) {
    return false;
  }
  while (Token->isOneOf(tok::semi, tok::comment)) {
    Token =
        Lexer::findNextToken(Token->getLocation(), Context.getSourceManager(),
                             Context.getLangOpts());
    if (!Token.has_value()) {
      return false;
    }
  }
  return HandlerMacroLocations.find(Token->getLocation()) !=
         HandlerMacroLocations.end();
}

void UnsafeKernelCallCheck::reportIssue(const Stmt &Stmt, ASTContext &Context) {
  // Get the wrapping expression
  const clang::Stmt *ExprWithCleanups =
      getParent<clang::Stmt, clang::ExprWithCleanups>(Stmt, Context);

  // Under certain compilation options kernel calls may not be wrapped
  // in cleanups
  if (!ExprWithCleanups) {
    ExprWithCleanups = &Stmt;
  }

  const bool IsInMacro = ExprWithCleanups->getBeginLoc().isInvalid() ||
                         ExprWithCleanups->getBeginLoc().isMacroID() ||
                         ExprWithCleanups->getEndLoc().isInvalid() ||
                         ExprWithCleanups->getEndLoc().isMacroID();

  if (!HandlerName.empty()) {
    const auto DiagnosticBuilder = diag(
        Stmt.getBeginLoc(), (llvm::Twine("Possible unchecked error after a "
                                       "kernel launch. Try adding the `") +
                           HandlerName + "()` macro after the kernel call:")
                              .str());
    if (IsInMacro) {
      return;
    }
    const auto ExprTerminator = utils::lexer::findNextTerminator(
        ExprWithCleanups->getEndLoc(), Context.getSourceManager(),
        Context.getLangOpts());
    const auto ParentStmt = getParent<clang::Stmt>(*ExprWithCleanups, Context);
    assert(ParentStmt);
    DiagnosticBuilder << utils::fixit::addSubsequentStatement(
        SourceRange(ExprWithCleanups->getBeginLoc(), ExprTerminator),
        *ParentStmt, HandlerName + "()", Context);
  } else {
    diag(Stmt.getBeginLoc(),
         "Possible unchecked error after a kernel launch. Try using "
         "`cudaGetLastError()` right after the kernel call to get the error or "
         "specify a project-wide kernel call error handler.");
  }
}

} // namespace cuda
} // namespace tidy
} // namespace clang
