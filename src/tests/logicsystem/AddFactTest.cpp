#include <gtest/gtest.h>
#include <Clause.h>
#include <KnowledgeBase.h>
#include <Fact.h>
#include <Resolution.h>
#include <ResolutionPair.h>



// TEST(ResolutionTest, ClauseWithVariableAndFactWithConstant) {
//     LogicSystem::KnowledgeBase kb;
    
//     // 添加谓词
//     int predicateId = kb.addPredicate("P");
    
//     // 添加常量和变量
//     auto constantId = kb.addVariable("a");
//     auto variableId = kb.addVariable("x");
    
//     // 创建一个含有常量的Fact: P(a)
//     LogicSystem::Clause clause1;
//     clause1.addLiteral({predicateId, {constantId}, false});
//     kb.addClause(clause1);
    
//     // 创建一个含有变量的Clause: ¬P(x)
//     LogicSystem::Clause clause2;
//     clause2.addLiteral({predicateId, {variableId}, true}); // true表示这是一个否定的文字
//     kb.addClause(clause2);
    
//     std::cout << "Clause1: " << clause1.toString(kb) << std::endl;
//     std::cout << "Clause2: " << clause2.toString(kb) << std::endl;
    
//     // 执行消解
//     LogicSystem::Resolution resolution;
//     auto result = resolution.testResolve(clause1, clause2, 0, 0, kb);
    
//     // 验证结果
//     ASSERT_TRUE(result.has_value());
//     EXPECT_TRUE(result->getLiterals().empty());
    
//     std::cout << "Resolution result: " << (result->getLiterals().empty() ? "Empty clause (contradiction found)" : result->toString(kb)) << std::endl;
// }