find_package(FLEX REQUIRED)

FLEX_TARGET(FOLScanner fol.l ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.cc)

add_library(lexer
    ${FLEX_FOLScanner_OUTPUTS}
    # 其他 lexer 相关的源文件
)

target_include_directories(lexer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(lexer ${FLEX_LIBRARIES})