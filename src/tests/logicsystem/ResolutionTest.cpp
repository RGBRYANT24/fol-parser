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
    // 创建两个子句: [P(x), G(x)] 和 [¬P(a), F(y)]
    Clause clause1;
    clause1.addLiteral(Literal(kb.getPredicateId("P").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
    clause1.addLiteral(Literal(kb.getPredicateId("G").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
    
    Clause clause2;
    clause2.addLiteral(Literal(kb.getPredicateId("P").value(), std::vector<SymbolId>{kb.getSymbolId("a").value()}, true));
    clause2.addLiteral(Literal(kb.getPredicateId("F").value(), std::vector<SymbolId>{kb.getSymbolId("y").value()}, false));

    // 执行消解
    Resolution resolver;
    std::optional<Clause> result = resolver.testResolve(clause2, clause1, 0, 0, kb);

    // 验证结果
    ASSERT_TRUE(result.has_value());
    std::cout << "After resolution [P(x), G(x)] and [¬P(a), F(y)]: " << result.value().toString(kb) << std::endl;

    // 检查结果子句中的文字数量
    ASSERT_EQ(result.value().getLiterals().size(), 2);

    // 检查子句是否包含 G(a) 和 F(y)
    bool hasGa = false;
    bool hasFy = false;

    for (const auto& literal : result.value().getLiterals()) {
        if (kb.getPredicateName(literal.getPredicateId()) == "G" &&
            literal.getArgumentIds().size() == 1 &&
            kb.getSymbolName(literal.getArgumentIds()[0]) == "a" &&
            !literal.isNegated()) {
            hasGa = true;
        }
        else if (kb.getPredicateName(literal.getPredicateId()) == "F" &&
                 literal.getArgumentIds().size() == 1 &&
                 kb.getSymbolName(literal.getArgumentIds()[0]) == "y" &&
                 !literal.isNegated()) {
            hasFy = true;
        }
    }

    EXPECT_FALSE(result.value().isTautology()) << "The clause is not Tautology";
    EXPECT_TRUE(hasGa) << "The clause should contain G(a)";
    EXPECT_TRUE(hasFy) << "The clause should contain F(y)";
}

} // namespace LogicSystem
