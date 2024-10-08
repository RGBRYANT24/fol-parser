#include <gtest/gtest.h>
#include "Clause.h"
#include "Literal.h"
#include "Resolution.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    class K3TwoColoringResolutionTest : public ::testing::Test
    {
    protected:
        KnowledgeBase kb;

        void SetUp() override
        {
            // 设置知识库，添加必要的符号
            kb.addConstant("a");
            kb.addConstant("b");
            kb.addConstant("c");
            kb.addVariable("x");
            kb.addVariable("y");
            kb.addPredicate("E");
            kb.addPredicate("R");
            kb.addPredicate("G");
            kb.addPredicate("B");
            kb.addPredicate("uncol");
        }
    };

    TEST_F(K3TwoColoringResolutionTest, K3GraphCannotBeThreeColored)
    {
        // 创建表示K3图三染色问题的子句

        // ¬E(x,y) ∨ ¬R(x) ∨ ¬R(y) ∨ uncol
        Clause adjacentNotSameColorR;
        adjacentNotSameColorR.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("x").value(), kb.getSymbolId("y").value()}, true));
        adjacentNotSameColorR.addLiteral(Literal(kb.getPredicateId("R").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, true));
        adjacentNotSameColorR.addLiteral(Literal(kb.getPredicateId("R").value(), std::vector<SymbolId>{kb.getSymbolId("y").value()}, true));
        adjacentNotSameColorR.addLiteral(Literal(kb.getPredicateId("uncol").value(), std::vector<SymbolId>{}, false));
        kb.addClause(adjacentNotSameColorR);
        // ¬E(x,y) ∨ ¬G(x) ∨ ¬G(y) ∨ uncol
        Clause adjacentNotSameColorG;
        adjacentNotSameColorG.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("x").value(), kb.getSymbolId("y").value()}, true));
        adjacentNotSameColorG.addLiteral(Literal(kb.getPredicateId("G").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, true));
        adjacentNotSameColorG.addLiteral(Literal(kb.getPredicateId("G").value(), std::vector<SymbolId>{kb.getSymbolId("y").value()}, true));
        adjacentNotSameColorG.addLiteral(Literal(kb.getPredicateId("uncol").value(), std::vector<SymbolId>{}, false));
        kb.addClause(adjacentNotSameColorG);
        // // ¬E(x,y) ∨ ¬B(x) ∨ ¬B(y) ∨ uncol
        // Clause adjacentNotSameColorB;
        // adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("x").value(), kb.getSymbolId("y").value()}, true));
        // adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, true));
        // adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("y").value()}, true));
        // adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("uncol").value(), std::vector<SymbolId>{}, false));
        // kb.addClause(adjacentNotSameColorB);
        // R(x) ∨ G(x) ∨ B(x)
        Clause vertexMustBeColored;
        vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("R").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("G").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        //vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        kb.addClause(vertexMustBeColored);
        // E(a,b)
        Clause edgeAB;
        edgeAB.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("a").value(), kb.getSymbolId("b").value()}, false));
        kb.addClause(edgeAB);
        // E(a,c)
        Clause edgeAC;
        edgeAC.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("a").value(), kb.getSymbolId("c").value()}, false));
        kb.addClause(edgeAC);
        // E(b,c)
        Clause edgeBC;
        edgeBC.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("b").value(), kb.getSymbolId("c").value()}, false));
        kb.addClause(edgeBC);
        // ¬uncol (goal)

        std::cout << "k3 2Colored KB " << std::endl;
        kb.print();
        Clause goal;
        goal.addLiteral(Literal(kb.getPredicateId("uncol").value(), std::vector<SymbolId>{}, true));
        // kb.addClause(goal);
        //  执行消解
        Resolution resolver;
        //std::vector<Clause> clauses = {adjacentNotSameColorR, adjacentNotSameColorG, adjacentNotSameColorB, vertexMustBeColored, edgeAB, edgeAC, edgeBC, goal};
        std::vector<Clause> clauses = {adjacentNotSameColorR, adjacentNotSameColorG, vertexMustBeColored, edgeAB, edgeAC, edgeBC, goal};

        bool proved = LogicSystem::Resolution::proveBFS(kb, goal);
        // 验证结果
        ASSERT_TRUE(proved);
        std::cout << "Resolution proves that K3 graph cannot be three-colored." << std::endl;
    }

} // namespace LogicSystem