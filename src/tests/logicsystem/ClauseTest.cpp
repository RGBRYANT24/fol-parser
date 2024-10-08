#include <gtest/gtest.h>
#include <Clause.h>
#include <KnowledgeBase.h>


TEST(ClauseTest, SomeTest) {
    LogicSystem::KnowledgeBase kb;
    // 添加谓词
    int dogId = kb.addPredicate("Dog");
    int animalId = kb.addPredicate("Animal");
    int breathesId = kb.addPredicate("Breathes");
    int ownsId = kb.addPredicate("Owns");

    // 添加常量
    LogicSystem::SymbolId xiaomingId = kb.addVariable("Xiaoming");

    // 添加变量
    LogicSystem::SymbolId xId = kb.addVariable("X");
    LogicSystem::SymbolId yId = kb.addVariable("Y");

    LogicSystem::Clause clause1;
    clause1.addLiteral(LogicSystem::Literal(dogId, {xId}, true));  // ¬Dog(X)
    clause1.addLiteral(LogicSystem::Literal(animalId, {xId}, true));  // ¬Animal(X)
    clause1.addLiteral(LogicSystem::Literal(animalId, {yId}, true));  // Animal(X)
    std::cout<< clause1.toString(kb) << std::endl;
}

// More tests...