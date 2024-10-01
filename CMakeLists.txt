cmake_policy(SET CMP0135 NEW)
cmake_minimum_required(VERSION 3.10)
project(FOL_Parser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Automatically find and include all subdirectories in the include folder
file(GLOB_RECURSE HEADER_FILES 
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/include/*.hpp
)

set(INCLUDE_DIRS "")
foreach(_headerFile ${HEADER_FILES})
    get_filename_component(_dir ${_headerFile} PATH)
    list(APPEND INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES INCLUDE_DIRS)

include_directories(${INCLUDE_DIRS})

# Add subdirectories for each module
add_subdirectory(src/ast)
add_subdirectory(src/lexer)
add_subdirectory(src/logicsystem)
add_subdirectory(src/parser)

# Create main executable
add_executable(fol_parser 
    ${FLEX_FOLScanner_OUTPUTS}
    ${BISON_FOLParser_OUTPUTS}
    src/main.cpp
)

# Link against module libraries and Flex/Bison libraries
target_link_libraries(fol_parser 
    ast
    lexer
    logicsystem
    parser
    ${FLEX_LIBRARIES}
    ${BISON_LIBRARIES}
)

# Enable testing
enable_testing()

# Add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test directory
add_subdirectory(src/tests)