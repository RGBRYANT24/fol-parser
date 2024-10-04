#include <gtest/gtest.h>
#include "Clause.h"
#include "Literal.h"
#include "Resolution.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    class ResolutionTest : public ::testing::Test
    {
    protected:
        KnowledgeBase kb;

        void SetUp() override
        {
            // 设置知识库，添加必要的符号
            kb.addConstant("a");
            kb.addConstant("b");
            kb.addVariable("x");
            kb.addVariable("y");
            kb.addPredicate("P");
            kb.addPredicate("Q");
            kb.addPredicate("G");
            kb.addPredicate("F");
        }
    };

    TEST_F(ResolutionTest, SimpleResolution)
    {
        // clause1.addLiteral(LogicSystem::Literal(dogId, {xId}, true));  // ¬Dog(X)
        //  创建两个子句: [P(x), G(x)] 和 [¬P(a), F(y)]
        Clause clause1;
        clause1.addLiteral(Literal(kb.getPredicateId("P").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        clause1.addLiteral(Literal(kb.getPredicateId("G").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        
        // clause1.addLiteral(Literal(kb.getPredicateId("P"), {kb.getSymbolId("x")}, false));
        Clause clause2;
        clause2.addLiteral(Literal(kb.getPredicateId("P").value(), std::vector<SymbolId>{kb.getSymbolId("a").value()}, true));
        clause2.addLiteral(Literal(kb.getPredicateId("F").value(), std::vector<SymbolId>{kb.getSymbolId("y").value()}, false));

        // 执行消解
        Resolution resolver;
        std::optional<Clause> result = resolver.testResolve(clause2, clause1, 0, 0, kb);

        // 验证结果
        ASSERT_TRUE(result.has_value());
        std::cout << "After resolution [P(x), G(x)] and [¬P(a), F(b)]" << result.value().toString(kb) << std::endl;
    }

    // TEST_F(ResolutionTest, NoResolution)
    // {
    //     // 创建两个不能消解的子句: {P(x)} 和 {Q(a)}
    //     Clause clause1({Literal(kb.getPredicateId("P"), {kb.getSymbolId("x")}, false)});
    //     Clause clause2({Literal(kb.getPredicateId("Q"), {kb.getSymbolId("a")}, false)});

    //     // 执行消解
    //     Resolution resolver;
    //     std::optional<Clause> result = resolver.testResolve(clause1, clause2, kb);

    //     // 验证结果
    //     ASSERT_FALSE(result.has_value()); // 应该没有消解结果
    // }

} // namespace LogicSystem
