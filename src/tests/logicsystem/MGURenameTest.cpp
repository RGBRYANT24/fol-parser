#include <gtest/gtest.h>
#include "KnowledgeBase.h"
#include "Unifier.h"

namespace LogicSystem
{

    class MGURenameTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // 设置知识库的基本谓词
            pred_E = kb.addPredicate("E");         // 边关系
            pred_R = kb.addPredicate("R");         // 红色
            pred_G = kb.addPredicate("G");         // 绿色
            pred_B = kb.addPredicate("B");         // 蓝色
            pred_uncol = kb.addPredicate("uncol"); // 不可着色标记

            // 添加常量（图的顶点）
            const_a = kb.addConstant("a");
            const_b = kb.addConstant("b");
            const_c = kb.addConstant("c");
            const_d = kb.addConstant("d");

            // 添加变量
            var_x = kb.addVariable("x");
            var_y = kb.addVariable("y");
        }

        KnowledgeBase kb;
        // 谓词ID
        int pred_E;
        int pred_R;
        int pred_G;
        int pred_B;
        int pred_uncol;
        // 常量符号
        SymbolId const_a;
        SymbolId const_b;
        SymbolId const_c;
        SymbolId const_d;
        // 变量符号
        SymbolId var_x;
        SymbolId var_y;
    };

    TEST_F(MGURenameTest, renametest)
    {
        // ¬E(x,y)
        Clause adjacentNotSameColorR;
        Literal lit1 = Literal(pred_E, {var_x, const_a}, true);
        adjacentNotSameColorR.addLiteral(lit1);
        kb.addClause(adjacentNotSameColorR);

        // E(x,y)
        Clause adjacentNotSameColorG;
        Literal lit2 = Literal(pred_E, {var_x, var_y}, false);
        adjacentNotSameColorG.addLiteral(lit2);
        kb.addClause(adjacentNotSameColorG);

        auto mgu = Unifier::findMGU(lit1, lit2, kb);
        if (mgu)
        {
            // 统一化成功
            Unifier::printSubstitution(*mgu, kb);
            std::cout << "After Unify " << std::endl;
            lit1 = Unifier::applySubstitutionToLiteral(lit1, *mgu, kb);
            lit2 = Unifier::applySubstitutionToLiteral(lit2, *mgu, kb);
            std::cout << lit1.toString(kb) << std::endl;
            std::cout << lit2.toString(kb) << std::endl;
        }
        else
        {
            // 统一化失败
            std::cout << "Unification failed" << std::endl;
        }

        // 二次配对进行MGU搜索，检测会不会因为变量重命名导致变量不断变长
        mgu = Unifier::findMGU(lit1, lit2, kb);
        if (mgu)
        {
            // 统一化成功
            Unifier::printSubstitution(*mgu, kb);
            std::cout << "After Unify " << std::endl;
            lit1 = Unifier::applySubstitutionToLiteral(lit1, *mgu, kb);
            lit2 = Unifier::applySubstitutionToLiteral(lit2, *mgu, kb);
            std::cout << lit1.toString(kb) << std::endl;
            std::cout << lit2.toString(kb) << std::endl;
        }
        else
        {
            // 统一化失败
            std::cout << "Unification failed" << std::endl;
        }
    }
    TEST_F(MGURenameTest, colorTest)
    {
        // ¬R(y)
        Clause clause1;
        Literal lit1 = Literal(pred_R, {var_y}, true);
        clause1.addLiteral(lit1);
        kb.addClause(clause1);

        // R(x)
        Clause clause2;
        Literal lit2 = Literal(pred_R, {var_x}, false);
        Literal lit3 = Literal(pred_G, {var_x}, false);
        clause2.addLiteral(lit2);
        clause2.addLiteral(lit3);
        kb.addClause(clause2);

        auto mgu = Unifier::findMGU(lit1, lit2, kb);
        if (mgu)
        {
            // 统一化成功
            Unifier::printSubstitution(*mgu, kb);
            std::cout << "After First Unify " << std::endl;
            lit1 = Unifier::applySubstitutionToLiteral(lit1, *mgu, kb);
            lit2 = Unifier::applySubstitutionToLiteral(lit2, *mgu, kb);
            lit3 = Unifier::applySubstitutionToLiteral(lit3, *mgu, kb);
            std::cout << lit1.toString(kb) << std::endl;
            std::cout << lit2.toString(kb) << std::endl;
            std::cout << lit3.toString(kb) << std::endl;
        }
        else
        {
            // 统一化失败
            std::cout << "First unification failed" << std::endl;
        }

        // 二次配对进行MGU搜索
        mgu = Unifier::findMGU(lit1, lit2, kb);
        if (mgu)
        {
            // 统一化成功
            Unifier::printSubstitution(*mgu, kb);
            std::cout << "After Second Unify " << std::endl;
            lit1 = Unifier::applySubstitutionToLiteral(lit1, *mgu, kb);
            lit2 = Unifier::applySubstitutionToLiteral(lit2, *mgu, kb);
            lit3 = Unifier::applySubstitutionToLiteral(lit3, *mgu, kb);
            std::cout << lit1.toString(kb) << std::endl;
            std::cout << lit2.toString(kb) << std::endl;
            std::cout << lit3.toString(kb) << std::endl;
        }
        else
        {
            // 统一化失败
            std::cout << "Second unification failed" << std::endl;
        }
    }

    // 可以添加更多测试用例，比如K4的三染色问题等

} // namespace LogicSystem