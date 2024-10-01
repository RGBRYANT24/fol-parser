add_library(ast
    ConstantNode.cpp
    ForallNode.cpp
    FunctionNode.cpp
    Node.cpp
    PredicateNode.cpp
    TermListNode.cpp
    UnaryOpNode.cpp
)

target_include_directories(ast PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})