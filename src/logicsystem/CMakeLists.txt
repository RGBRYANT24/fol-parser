add_library(logicsystem
    Clause.cpp
    CNF.cpp
    ConstantTable.cpp
    Fact.cpp
    KnowledgeBase.cpp
    PredicateTable.cpp
    Resolution.cpp
    Unifier.cpp
    VariableTable.cpp
    Literal.cpp
    SLITree.cpp
    SLINode.cpp
    SLIResolution.cpp
    VariableRenamer.cpp
    SLIMCTSProver.cpp
    # KnowledgeBaseBuilder.cpp
)

add_subdirectory(operation)

target_include_directories(logicsystem
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    # ${CMAKE_SOURCE_DIR}/src/lexer
    # ${CMAKE_SOURCE_DIR}/src/parser
)



# # 链接 parser_lib（它会自动包含 lexer_lib）
# target_link_libraries(logicsystem PUBLIC lexer parser)
# # 获取目标属性
# get_target_property(LEXER_TYPE lexer TYPE)
# get_target_property(PARSER_TYPE parser TYPE)

# message(STATUS "Lexer library type: ${LEXER_TYPE}")
# message(STATUS "Parser library type: ${PARSER_TYPE}")