# Operation/CMakeLists.txt

# 收集当前目录下的所有源文件
set(OPERATION_SOURCES
    AddOperation.cpp
    Operation.cpp
    TruncateOperation.cpp
)

# 创建一个静态库
add_library(operation STATIC
    ${OPERATION_SOURCES}
)

# 指定头文件包含目录
target_include_directories(operation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # 当前目录
        ${CMAKE_SOURCE_DIR}/src      # 项目根源码目录
)

# 如果operation库依赖于其他库，需要链接它们
target_link_libraries(operation
    PUBLIC
        # 这里添加其他依赖的库
)