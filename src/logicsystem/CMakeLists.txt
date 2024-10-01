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
)

target_include_directories(logicsystem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
