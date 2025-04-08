cmake_policy(SET CMP0135 NEW)
cmake_minimum_required(VERSION 3.10)
project(FOL_Parser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

# Set output directory relative to the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

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
include_directories(${CMAKE_SOURCE_DIR}/third_party)
# Add subdirectories for each module
add_subdirectory(src/ast)
# 暂时注释掉parser和lexer
#add_subdirectory(src/parser)
#add_subdirectory(src/lexer)
add_subdirectory(src/logicsystem)

# Create main executable
add_executable(fol_parser 
    src/main.cpp
)

# Link against module libraries
target_link_libraries(fol_parser 
    ast
    #lexer
    #parser
    logicsystem
    #graphcheckers
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