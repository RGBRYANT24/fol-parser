#include <gtest/gtest.h>
#include "Clause.h"
#include "Literal.h"
#include "Resolution.h"
#include "KnowledgeBase.h"
#include "BFSStrategy.h"

namespace LogicSystem
{
    class GraphColoringResolutionTest : public ::testing::Test
    {
    protected:
        KnowledgeBase kb;

        void SetUp() override
        {
            // 设置知识库，添加必要的符号
            kb.addConstant("a");
            kb.addConstant("b");
            kb.addConstant("c");
            kb.addConstant("d");
            kb.addVariable("x");
            kb.addVariable("y");
            kb.addPredicate("E");
            kb.addPredicate("R");
            kb.addPredicate("G");
            kb.addPredicate("B");
            kb.addPredicate("uncol");
        }
    };

    TEST_F(GraphColoringResolutionTest, K3GraphCannotBeTwoColored)
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
        // vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
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
        // std::vector<Clause> clauses = {adjacentNotSameColorR, adjacentNotSameColorG, adjacentNotSameColorB, vertexMustBeColored, edgeAB, edgeAC, edgeBC, goal};
        std::vector<Clause> clauses = {adjacentNotSameColorR, adjacentNotSameColorG, vertexMustBeColored, edgeAB, edgeAC, edgeBC, goal};
        BFSStrategy bfsStrategy;
        bool proved = LogicSystem::Resolution::prove(kb, goal,bfsStrategy);
        // 验证结果
        ASSERT_TRUE(proved);
        std::cout << "Resolution proves that K3 graph cannot be three-colored." << std::endl;
    }

    TEST_F(GraphColoringResolutionTest, K4GraphCannotBeThreeColored)
    {
        // 创建表示K4图三染色问题的子句

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

        // ¬E(x,y) ∨ ¬B(x) ∨ ¬B(y) ∨ uncol
        Clause adjacentNotSameColorB;
        adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("E").value(), std::vector<SymbolId>{kb.getSymbolId("x").value(), kb.getSymbolId("y").value()}, true));
        adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, true));
        adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("y").value()}, true));
        adjacentNotSameColorB.addLiteral(Literal(kb.getPredicateId("uncol").value(), std::vector<SymbolId>{}, false));
        kb.addClause(adjacentNotSameColorB);

        // R(x) ∨ G(x) ∨ B(x)
        Clause vertexMustBeColored;
        vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("R").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("G").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        vertexMustBeColored.addLiteral(Literal(kb.getPredicateId("B").value(), std::vector<SymbolId>{kb.getSymbolId("x").value()}, false));
        kb.addClause(vertexMustBeColored);

        // 添加K4图的所有边
        std::vector<std::pair<std::string, std::string>> edges = {
            {"a", "b"}, {"a", "c"}, {"a", "d"}, {"b", "c"}, {"b", "d"}, {"c", "d"}};

        for (const auto &edge : edges)
        {
            Clause edgeClause1, edgeClause2;
            edgeClause1.addLiteral(Literal(kb.getPredicateId("E").value(),
                                          std::vector<SymbolId>{kb.getSymbolId(edge.first).value(), kb.getSymbolId(edge.second).value()}, false));
            edgeClause2.addLiteral(Literal(kb.getPredicateId("E").value(),
                                           std::vector<SymbolId>{kb.getSymbolId(edge.second).value(), kb.getSymbolId(edge.first).value()}, false));
            kb.addClause(edgeClause1);
            kb.addClause(edgeClause2);
        }

        std::cout << "K4 3Colored KB " << std::endl;
        kb.print();

        // 设置目标: ¬uncol
        Clause goal;
        goal.addLiteral(Literal(kb.getPredicateId("uncol").value(), std::vector<SymbolId>{}, true));

        // // 执行消解
        BFSStrategy bfsStrategy;
        bool proved = LogicSystem::Resolution::prove(kb, goal,bfsStrategy);

        // 验证结果
        ASSERT_TRUE(proved);
        std::cout << "Resolution proves that K4 graph cannot be three-colored." << std::endl;
    }

} // namespace LogicSystem