add_executable(logicsystem_tests
    # 列出该模块的测试源文件
    ClauseTest.cpp
    AddFactTest.cpp
    KBTest.cpp
    ResolutionTest.cpp
    # 2ColorableTest.cpp
    SLITreeTest.cpp
    UnifierTest.cpp
    SLIResolutionTest.cpp
    ColoringSLIResolutionTest
    MGURenameTest.cpp
    LiteralHashTest.cpp
    SLITreeHashTest.cpp
    VariableRenamerTest.cpp
    ColoringSLIMCTSTest.cpp
    SLIMCTSDeepCopyTest.cpp
    RandomGraphColoringSLIMCTSTest.cpp
    # KnowledgeBaseBuilderTest.cpp
)
# 对于测试可执行文件
set_target_properties(logicsystem_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/logicsystem
)

target_link_libraries(logicsystem_tests
    PUBLIC
    gtest_main
    # lexer
    # parser
    logicsystem
)