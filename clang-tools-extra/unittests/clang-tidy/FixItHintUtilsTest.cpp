#include "utils/FixItHintUtils.h"
#include "ClangTidyCheck.h"
#include "ClangTidyTest.h"
#include "utils/LexerUtils.h"
#include "gtest/gtest.h"

#define REGISTER_TEST_MATCHER(TestSuiteName, MatcherType)                      \
  class TestSuiteName : public ClangTidyCheck {                                \
  public:                                                                      \
    TestSuiteName(llvm::StringRef Name,                                        \
                  clang::tidy::ClangTidyContext *Context)                      \
        : ClangTidyCheck(Name, Context) {}                                     \
    void registerMatchers(clang::ast_matchers::MatchFinder *Finder) override { \
      Finder->addMatcher(getMatcher(), this);                                  \
    }                                                                          \
    void check(                                                                \
        const clang::ast_matchers::MatchFinder::MatchResult &Result) override; \
                                                                               \
  private:                                                                     \
    clang::ast_matchers::internal::Matcher<MatcherType> getMatcher();          \
  };                                                                           \
  clang::ast_matchers::internal::Matcher<MatcherType>                          \
  TestSuiteName::getMatcher()

#define CHECK_TEST_MATCHER(TestSuiteName)                                      \
  void TestSuiteName::check(                                                   \
      const clang::ast_matchers::MatchFinder::MatchResult &Result)

#define RUN_TEST_MATCHER(TestSuiteName, TestName, SourceCode, TargetCode)      \
  TEST(TestSuiteName, TestName) {                                              \
    EXPECT_EQ(TargetCode, runCheckOnCode<TestSuiteName>(SourceCode));          \
  }

namespace clang {
namespace tidy {
namespace test {

using namespace ast_matchers;

REGISTER_TEST_MATCHER(AddSubsequentStatementUtil, Stmt) {
  return stmt(forEach(callExpr().bind("call"))).bind("parent");
}

CHECK_TEST_MATCHER(AddSubsequentStatementUtil) {
  const Stmt *const Node = Result.Nodes.getNodeAs<Stmt>("call");
  const Stmt *const ParentNode = Result.Nodes.getNodeAs<Stmt>("parent");
  EXPECT_TRUE(Node);
  EXPECT_TRUE(ParentNode);
  const auto NodeTerminator = utils::lexer::findNextTerminator(
      Node->getEndLoc(), Result.Context->getSourceManager(),
      Result.Context->getLangOpts());
  auto Range = SourceRange(Node->getBeginLoc(), NodeTerminator);
  diag(Node->getBeginLoc(), "") << utils::fixit::addSubsequentStatement(
      Range, *ParentNode, "foo()", *Result.Context);
}

RUN_TEST_MATCHER(AddSubsequentStatementUtil, CompoundStatementParent,
                 "void foo() {\n  foo();\n}",
                 "void foo() {\n  foo();\n  foo();\n}")

RUN_TEST_MATCHER(AddSubsequentStatementUtil,
                 CompoundStatementParentWithComments,
                 "void foo() {\n \tfoo(); //some /* comments */\n}",
                 "void foo() {\n \tfoo(); //some /* comments */\n \tfoo();\n}")

RUN_TEST_MATCHER(AddSubsequentStatementUtil, IfStatementParent,
                 "void foo() {\n  if(true)\n    foo();\n}",
                 "void foo() {\n  if(true) {\n    foo();\n    foo();\n  }\n}")

RUN_TEST_MATCHER(
    AddSubsequentStatementUtil, IfStatementParentWithComments,
    "void foo() {\n  if(true)\n    foo(); //some /* comments */\n}",
    "void foo() {\n  if(true) {\n    foo(); //some /* comments */\n    "
    "foo();\n  }\n}")

RUN_TEST_MATCHER(
    AddSubsequentStatementUtil, IfStatementSameLineParent,
    "void foo() {\n  if(true) foo();\n}",
    "void foo() {\n  if(true) { foo();\n             foo();\n  }\n}")

RUN_TEST_MATCHER(AddSubsequentStatementUtil,
                 IfStatementSameLineParentWithComments,
                 "void foo() {\n  if(true) foo(); //some /* comments */\n}",
                 "void foo() {\n  if(true) { foo(); //some /* comments */\n    "
                 "         foo();\n  }\n}")

RUN_TEST_MATCHER(AddSubsequentStatementUtil, IfElseStatementParent,
                 "void foo() {\n  if(true)\n    foo();\n  else\n    foo();\n}",
                 "void foo() {\n  if(true) {\n    foo();\n    foo();\n  } else "
                 "{\n    foo();\n    foo();\n  }\n}")

RUN_TEST_MATCHER(
    AddSubsequentStatementUtil, IfElseStatementParentWithComments,
    "void foo() {\n  if(true)\n    foo(); //some /* comments */\n  else\n    "
    "foo(); //some /* comments */\n}",
    "void foo() {\n  if(true) {\n    foo(); //some /* comments */\n    "
    "foo();\n  } else {\n    foo(); //some /* comments */\n    foo();\n  }\n}")

} // namespace test
} // namespace tidy
} // namespace clang
