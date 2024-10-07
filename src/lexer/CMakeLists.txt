find_package(FLEX REQUIRED)

FLEX_TARGET(FOLScanner fol.l ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.cc)

# 创建 lexer_lib（注意：我们将库名称改为 lexer_lib 以避免与目录名冲突）
add_library(lexer STATIC
    ${FLEX_FOLScanner_OUTPUTS}
    # 其他 lexer 相关的源文件（如果有的话）
)

target_include_directories(lexer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}    
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/../parser
    ${CMAKE_SOURCE_DIR}/include  # 添加全局 include 目录
)
target_link_libraries(lexer ${FLEX_LIBRARIES})

# 导出 lexer_lib，使其他目录可以使用
export(TARGETS lexer FILE ${CMAKE_BINARY_DIR}/lexer.cmake)