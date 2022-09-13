#include "utils/Matchers.h"
#include "../../clang/unittests/ASTMatchers/ASTMatchersTest.h"

namespace clang {
namespace tidy {
namespace test {

using namespace ast_matchers;

TEST_P(ASTMatchersTest, isValueUnused) {
  auto Matcher = matchers::isValueUnused(integerLiteral(equals(42)));
  std::string CodePrefix = "void bar()";

  matches(CodePrefix + "{42;}", Matcher);
  matches(CodePrefix + "{int x = ({42; 0;});}", Matcher);
  matches(CodePrefix + "{if (true) 42;}", Matcher);
  matches(CodePrefix + "{while(true) 42;}", Matcher);
  matches(CodePrefix + "{do 42; while(true);}", Matcher);
  matches(CodePrefix + "{for(;;) 42;}", Matcher);
  matches(CodePrefix + "{for(42;;) bar();}", Matcher);
  matches(CodePrefix + "{for(;;42) bar();}", Matcher);
  matches(CodePrefix + "{int t[] = {1, 2, 3}; for(int x : t) 42;}", Matcher);
  matches(CodePrefix + "{switch(1) {case 42:}", Matcher);

  notMatches(CodePrefix + "{bar();}", Matcher);
  notMatches(CodePrefix + "{int x = 42;}", Matcher);
  notMatches(CodePrefix + "{int x = ({42;});}", Matcher);
  notMatches(CodePrefix + "{if (42) bar();}", Matcher);
  notMatches(CodePrefix + "{while(42) bar();}", Matcher);
  notMatches(CodePrefix + "{do bar(); while(42);}", Matcher);
  notMatches(CodePrefix + "{for(; 42; )} bar();", Matcher);
  notMatches(CodePrefix + "switch(1) {default: bar();}", Matcher);
}

} // namespace test
} // namespace tidy
} // namespace clang
