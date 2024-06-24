cmake_minimum_required(VERSION 3.10)
project(FOL_Parser)

set(CMAKE_CXX_STANDARD 14)

# Set output directory relative to the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Find Flex and Bison
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Generate lexer and parser
FLEX_TARGET(FOLScanner ${CMAKE_SOURCE_DIR}/src/lexer/fol.l ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.cc)
BISON_TARGET(FOLParser ${CMAKE_SOURCE_DIR}/src/parser/fol.y ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.cc
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/fol.tab.hh)
ADD_FLEX_BISON_DEPENDENCY(FOLScanner FOLParser)

# Include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all source files
file(GLOB_RECURSE AST_SOURCES "${CMAKE_SOURCE_DIR}/src/ast/*.cpp")

# Create executable
add_executable(fol_parser 
    ${FLEX_FOLScanner_OUTPUTS}
    ${BISON_FOLParser_OUTPUTS}
    ${AST_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

# Link against flex and bison libraries if needed
target_link_libraries(fol_parser ${FLEX_LIBRARIES} ${BISON_LIBRARIES})