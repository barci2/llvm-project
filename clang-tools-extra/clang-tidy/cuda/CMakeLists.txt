add_clang_library(clangTidyCudaModule
  CudaTidyModule.cpp
  UnsafeApiCallCheck.cpp
  UnsafeKernelCallCheck.cpp
  LINK_LIBS
  clangTidy
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyAlteraModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
