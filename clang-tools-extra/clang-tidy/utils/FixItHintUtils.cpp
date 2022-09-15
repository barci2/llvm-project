//===--- FixItHintUtils.cpp - clang-tidy-----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FixItHintUtils.h"
#include "LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

namespace clang {
namespace tidy {
namespace utils {
namespace fixit {

FixItHint changeVarDeclToReference(const VarDecl &Var, ASTContext &Context) {
  SourceLocation AmpLocation = Var.getLocation();
  auto Token = utils::lexer::getPreviousToken(
      AmpLocation, Context.getSourceManager(), Context.getLangOpts());
  if (!Token.is(tok::unknown))
    AmpLocation = Lexer::getLocForEndOfToken(Token.getLocation(), 0,
                                             Context.getSourceManager(),
                                             Context.getLangOpts());
  return FixItHint::CreateInsertion(AmpLocation, "&");
}

static bool isValueType(const Type *T) {
  return !(isa<PointerType>(T) || isa<ReferenceType>(T) || isa<ArrayType>(T) ||
           isa<MemberPointerType>(T) || isa<ObjCObjectPointerType>(T));
}
static bool isValueType(QualType QT) { return isValueType(QT.getTypePtr()); }
static bool isMemberOrFunctionPointer(QualType QT) {
  return (QT->isPointerType() && QT->isFunctionPointerType()) ||
         isa<MemberPointerType>(QT.getTypePtr());
}

static bool locDangerous(SourceLocation S) {
  return S.isInvalid() || S.isMacroID();
}

static Optional<SourceLocation>
skipLParensBackwards(SourceLocation Start, const ASTContext &Context) {
  if (locDangerous(Start))
    return None;

  auto PreviousTokenLParen = [&Start, &Context]() {
    Token T;
    T = lexer::getPreviousToken(Start, Context.getSourceManager(),
                                Context.getLangOpts());
    return T.is(tok::l_paren);
  };

  while (Start.isValid() && PreviousTokenLParen())
    Start = lexer::findPreviousTokenStart(Start, Context.getSourceManager(),
                                          Context.getLangOpts());

  if (locDangerous(Start))
    return None;
  return Start;
}

static Optional<FixItHint> fixIfNotDangerous(SourceLocation Loc,
                                             StringRef Text) {
  if (locDangerous(Loc))
    return None;
  return FixItHint::CreateInsertion(Loc, Text);
}

// Build a string that can be emitted as FixIt with either a space in before
// or after the qualifier, either ' const' or 'const '.
static std::string buildQualifier(DeclSpec::TQ Qualifier,
                                  bool WhitespaceBefore = false) {
  if (WhitespaceBefore)
    return (llvm::Twine(' ') + DeclSpec::getSpecifierName(Qualifier)).str();
  return (llvm::Twine(DeclSpec::getSpecifierName(Qualifier)) + " ").str();
}

static Optional<FixItHint> changeValue(const VarDecl &Var,
                                       DeclSpec::TQ Qualifier,
                                       QualifierTarget QualTarget,
                                       QualifierPolicy QualPolicy,
                                       const ASTContext &Context) {
  switch (QualPolicy) {
  case QualifierPolicy::Left:
    return fixIfNotDangerous(Var.getTypeSpecStartLoc(),
                             buildQualifier(Qualifier));
  case QualifierPolicy::Right:
    Optional<SourceLocation> IgnoredParens =
        skipLParensBackwards(Var.getLocation(), Context);

    if (IgnoredParens)
      return fixIfNotDangerous(*IgnoredParens, buildQualifier(Qualifier));
    return None;
  }
  llvm_unreachable("Unknown QualifierPolicy enum");
}

static Optional<FixItHint> changePointerItself(const VarDecl &Var,
                                               DeclSpec::TQ Qualifier,
                                               const ASTContext &Context) {
  if (locDangerous(Var.getLocation()))
    return None;

  Optional<SourceLocation> IgnoredParens =
      skipLParensBackwards(Var.getLocation(), Context);
  if (IgnoredParens)
    return fixIfNotDangerous(*IgnoredParens, buildQualifier(Qualifier));
  return None;
}

static Optional<FixItHint>
changePointer(const VarDecl &Var, DeclSpec::TQ Qualifier, const Type *Pointee,
              QualifierTarget QualTarget, QualifierPolicy QualPolicy,
              const ASTContext &Context) {
  // The pointer itself shall be marked as `const`. This is always to the right
  // of the '*' or in front of the identifier.
  if (QualTarget == QualifierTarget::Value)
    return changePointerItself(Var, Qualifier, Context);

  // Mark the pointee `const` that is a normal value (`int* p = nullptr;`).
  if (QualTarget == QualifierTarget::Pointee && isValueType(Pointee)) {
    // Adding the `const` on the left side is just the beginning of the type
    // specification. (`const int* p = nullptr;`)
    if (QualPolicy == QualifierPolicy::Left)
      return fixIfNotDangerous(Var.getTypeSpecStartLoc(),
                               buildQualifier(Qualifier));

    // Adding the `const` on the right side of the value type requires finding
    // the `*` token and placing the `const` left of it.
    // (`int const* p = nullptr;`)
    if (QualPolicy == QualifierPolicy::Right) {
      SourceLocation BeforeStar = lexer::findPreviousTokenKind(
          Var.getLocation(), Context.getSourceManager(), Context.getLangOpts(),
          tok::star);
      if (locDangerous(BeforeStar))
        return None;

      Optional<SourceLocation> IgnoredParens =
          skipLParensBackwards(BeforeStar, Context);

      if (IgnoredParens)
        return fixIfNotDangerous(*IgnoredParens,
                                 buildQualifier(Qualifier, true));
      return None;
    }
  }

  if (QualTarget == QualifierTarget::Pointee && Pointee->isPointerType()) {
    // Adding the `const` to the pointee if the pointee is a pointer
    // is the same as 'QualPolicy == Right && isValueType(Pointee)'.
    // The `const` must be left of the last `*` token.
    // (`int * const* p = nullptr;`)
    SourceLocation BeforeStar = lexer::findPreviousTokenKind(
        Var.getLocation(), Context.getSourceManager(), Context.getLangOpts(),
        tok::star);
    return fixIfNotDangerous(BeforeStar, buildQualifier(Qualifier, true));
  }

  return None;
}

static Optional<FixItHint>
changeReferencee(const VarDecl &Var, DeclSpec::TQ Qualifier, QualType Pointee,
                 QualifierTarget QualTarget, QualifierPolicy QualPolicy,
                 const ASTContext &Context) {
  if (QualPolicy == QualifierPolicy::Left && isValueType(Pointee))
    return fixIfNotDangerous(Var.getTypeSpecStartLoc(),
                             buildQualifier(Qualifier));

  SourceLocation BeforeRef = lexer::findPreviousAnyTokenKind(
      Var.getLocation(), Context.getSourceManager(), Context.getLangOpts(),
      tok::amp, tok::ampamp);
  Optional<SourceLocation> IgnoredParens =
      skipLParensBackwards(BeforeRef, Context);
  if (IgnoredParens)
    return fixIfNotDangerous(*IgnoredParens, buildQualifier(Qualifier, true));

  return None;
}

Optional<FixItHint> addQualifierToVarDecl(const VarDecl &Var,
                                          const ASTContext &Context,
                                          DeclSpec::TQ Qualifier,
                                          QualifierTarget QualTarget,
                                          QualifierPolicy QualPolicy) {
  assert((QualPolicy == QualifierPolicy::Left ||
          QualPolicy == QualifierPolicy::Right) &&
         "Unexpected Insertion Policy");
  assert((QualTarget == QualifierTarget::Pointee ||
          QualTarget == QualifierTarget::Value) &&
         "Unexpected Target");

  QualType ParenStrippedType = Var.getType().IgnoreParens();
  if (isValueType(ParenStrippedType))
    return changeValue(Var, Qualifier, QualTarget, QualPolicy, Context);

  if (ParenStrippedType->isReferenceType())
    return changeReferencee(Var, Qualifier, Var.getType()->getPointeeType(),
                            QualTarget, QualPolicy, Context);

  if (isMemberOrFunctionPointer(ParenStrippedType))
    return changePointerItself(Var, Qualifier, Context);

  if (ParenStrippedType->isPointerType())
    return changePointer(Var, Qualifier,
                         ParenStrippedType->getPointeeType().getTypePtr(),
                         QualTarget, QualPolicy, Context);

  if (ParenStrippedType->isArrayType()) {
    const Type *AT = ParenStrippedType->getBaseElementTypeUnsafe();
    assert(AT && "Did not retrieve array element type for an array.");

    if (isValueType(AT))
      return changeValue(Var, Qualifier, QualTarget, QualPolicy, Context);

    if (AT->isPointerType())
      return changePointer(Var, Qualifier, AT->getPointeeType().getTypePtr(),
                           QualTarget, QualPolicy, Context);
  }

  return None;
}

static unsigned int getLineNumber(SourceLocation Loc, SourceManager& SM) {
  FileID FID;
  unsigned int Offset;
  std::tie(FID, Offset) = SM.getDecomposedLoc(Loc);
  return SM.getLineNumber(FID, Offset);
}

static std::string getIndent(SourceLocation sLoc, ASTContext& context) {
  auto& SM = context.getSourceManager();

  const auto sLocLineNo = getLineNumber(sLoc, SM);

  auto indentation_template = tooling::fixit::internal::getText(
      CharSourceRange::getCharRange(SourceRange(
          SM.translateLineCol(SM.getFileID(sLoc), sLocLineNo, 1), sLoc)),
      context);

  std::string indentation;
  indentation.reserve(indentation_template.size());
  std::transform(
      indentation_template.begin(),
      indentation_template.end(),
      std::back_inserter(indentation),
      [](char c) { return isspace(c) ? c : ' '; });
  return indentation;
}

llvm::SmallVector<FixItHint, 1> addSubsequentStatement(
    SourceRange stmtRangeWithTerminator,
    const Stmt& parentStmt,
    llvm::StringRef nextStmt,
    ASTContext& context) {
  auto& SM = context.getSourceManager();
  auto langOpts = context.getLangOpts();

  const auto stmtEndLineNo =
      getLineNumber(stmtRangeWithTerminator.getEnd(), SM);

  // Find the first token's data for which the next token is
  // either a line apart or is not a comment
  SourceLocation lastTokenEndLoc =
      stmtRangeWithTerminator.getEnd().getLocWithOffset(1);
  auto lastTokenLine = stmtEndLineNo;
  bool insertNewLine = true;
  while (true) {
    llvm::Optional<Token> tokenOption = Lexer::findNextToken(
        lastTokenEndLoc.getLocWithOffset(-1), SM, langOpts, true);
    if (!tokenOption) {
      return llvm::SmallVector<FixItHint, 1>();
    }
    if (tokenOption->is(tok::eof)) {
      insertNewLine = false;
      break;
    }
    const auto tokenBeginLineNo = getLineNumber(tokenOption->getLocation(), SM);

    if (tokenOption->isNot(tok::comment)) {
      insertNewLine = tokenBeginLineNo != stmtEndLineNo;
      break;
    }
    if (tokenBeginLineNo > lastTokenLine) {
      break;
    }

    lastTokenEndLoc = tokenOption->getEndLoc();
    lastTokenLine = getLineNumber(tokenOption->getEndLoc(), SM);
  }

  bool isEnclosedWithBrackets =
      parentStmt.getStmtClass() == Stmt::CompoundStmtClass;

  // Generating the FixItHint
  // There are 5 scenarios that we have to take into account:
  // 1. The statement is enclosed in brackets but the next statement is
  //    in the same line - insert the new statement right after the previous one
  // 2. The statement is not enclosed in brackets and the next statement is
  //    in the same line - same as 1. and enclose both statements in brackets
  //    on the same line
  // 3. The statement is enclosed in brackets and the next statement is
  //    on subsequent lines - skip all the comments in this line
  // 4. The statement is not enclosed in brackets but the next statement is on
  //    subsequent lines - same as 3. and enclose the statements with
  //    google-style multiline brackets (opening bracket right after the parent
  //    statement and closing bracket on a new line after the new statement).
  // 5. The statement is not enclosed in brackets but the next statement is on
  //    subsequent lines and the main statement is before an else token - same
  //    as 4. but the closing bracket is put on the same line as the else
  //    statement
  if (!insertNewLine) {
    if (isEnclosedWithBrackets) {
      // Case 1.
      return llvm::SmallVector<FixItHint, 1>{FixItHint::CreateInsertion(
          stmtRangeWithTerminator.getEnd().getLocWithOffset(1),
          (llvm::Twine(" ") + nextStmt.str() + ";").str())};
    } else {
      // Case 2.
      return llvm::SmallVector<FixItHint, 1>{
          FixItHint::CreateInsertion(stmtRangeWithTerminator.getBegin(), "{"),
          FixItHint::CreateInsertion(
              stmtRangeWithTerminator.getEnd().getLocWithOffset(1),
              (llvm::Twine(" ") + nextStmt.str() + ";}").str())};
    }
  } else {
    if (isEnclosedWithBrackets) {
      // Case 3.
      return llvm::SmallVector<FixItHint, 1>{FixItHint::CreateInsertion(
          lastTokenEndLoc,
          (llvm::Twine("\n") +
           getIndent(stmtRangeWithTerminator.getBegin(), context) +
           nextStmt.str() + ";")
              .str())};
    } else {
      const auto previousTokenEndLoc =
          tidy::utils::lexer::getPreviousToken(
              stmtRangeWithTerminator.getBegin(), SM, context.getLangOpts())
              .getEndLoc();
      auto nextStmtIndent =
          getIndent(stmtRangeWithTerminator.getBegin(), context);

      if (getLineNumber(previousTokenEndLoc, SM) ==
          getLineNumber(stmtRangeWithTerminator.getBegin(), SM)) {
        nextStmtIndent += "  ";
      }
      auto nextToken = Lexer::findNextToken(lastTokenEndLoc, SM, langOpts);
      if (!nextToken || nextToken->getRawIdentifier() != "else") {
        // Case 4.
        return llvm::SmallVector<FixItHint, 1>{
            FixItHint::CreateInsertion(previousTokenEndLoc, " {"),
            FixItHint::CreateInsertion(
                lastTokenEndLoc,
                (llvm::Twine("\n") + nextStmtIndent + nextStmt.str() + ";\n" +
                 getIndent(parentStmt.getBeginLoc(), context) + "}")
                    .str())};
      } else {
        // Case 5.
        return llvm::SmallVector<FixItHint, 1>{
            FixItHint::CreateInsertion(previousTokenEndLoc, " {"),
            FixItHint::CreateInsertion(
                lastTokenEndLoc,
                (llvm::Twine("\n") + nextStmtIndent + nextStmt.str() + ";")
                    .str()),
            FixItHint::CreateInsertion(nextToken->getLocation(), "} ")};
      }
    }
  }
}

} // namespace fixit
} // namespace utils
} // namespace tidy
} // namespace clang
