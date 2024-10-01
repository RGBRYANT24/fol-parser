find_package(BISON REQUIRED)

BISON_TARGET(FOLParser fol.y ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.cc
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.hh)

add_library(parser
    ${BISON_FOLParser_OUTPUTS}
    # 其他 parser 相关的源文件
)

target_include_directories(parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(parser ${BISON_LIBRARIES})