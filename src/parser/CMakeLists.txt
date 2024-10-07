find_package(BISON REQUIRED)

BISON_TARGET(FOLParser fol.y ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.cc
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.hh)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Expected fol.tab.hh location: ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.hh")
message(STATUS "Bison output: ${BISON_FOLParser_OUTPUTS}")


# 创建 parser
add_library(parser STATIC
    ${BISON_FOLParser_OUTPUTS}
    # 其他 parser 相关的源文件（如果有的话）
)

# add_dependencies(parser ${BISON_FOLParser_OUTPUTS})

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/fol.tab.hh")
  message(STATUS "fol.tab.hh has been generated successfully.")
else()
  message(WARNING "fol.tab.hh has not been generated!")
endif()

# 设置 include 目录
target_include_directories(parser PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
    # ${CMAKE_CURRENT_BINARY_DIR}/../lexer  # 添加 lexer 目录
)


# 链接 Bison 库 lexer ast
target_link_libraries(parser PUBLIC
    ${BISON_LIBRARIES}
    lexer
    ast
)

# # 添加 Flex 和 Bison 的依赖关系
# add_dependencies(parser_lib ${FLEX_FOLScanner_TARGET})

# 导出 parser_lib，使其他目录可以使用
export(TARGETS parser ast FILE ${CMAKE_BINARY_DIR}/parser.cmake)

# # 添加自定义命令来检查生成的文件
# add_custom_command(
#   TARGET parser
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E echo "Checking for fol.tab.hh..."
#   COMMAND ${CMAKE_COMMAND} -E if_exists "${CMAKE_CURRENT_BINARY_DIR}/fol.tab.hh" "${CMAKE_COMMAND}" -E echo "fol.tab.hh exists" "${CMAKE_COMMAND}" -E echo "fol.tab.hh does not exist"
# )

# # 输出 Bison 命令
# message(STATUS "Bison command: ${BISON_EXECUTABLE} ${BISON_FOLParser_COMPILE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.cc ${CMAKE_CURRENT_SOURCE_DIR}/fol.y")

# # 检查 fol.y 文件是否存在
# if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fol.y")
#   message(FATAL_ERROR "fol.y file not found in ${CMAKE_CURRENT_SOURCE_DIR}")
# endif()