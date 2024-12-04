#include <gtest/gtest.h>
#include "Unifier.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{

    class UnifierTest : public ::testing::Test
    {
    protected:
        KnowledgeBase kb;
        int pred_P;
        int pred_Q;
        SymbolId const_a;
        SymbolId const_b;
        SymbolId var_x;
        SymbolId var_y;
        void SetUp() override
        {
            // 设置知识库
            pred_P = kb.addPredicate("P"); // 二元谓词 P
            pred_Q = kb.addPredicate("Q"); // 一元谓词 Q
            const_a = kb.addConstant("a");
            const_b = kb.addConstant("b");
            var_x = kb.addVariable("x");
            var_y = kb.addVariable("y");
        }
    };

    TEST_F(UnifierTest, SimpleUnification)
    {
        // p(X, a) 和 p(b, y) 的统一
        // P(x,a)
        Literal l1(pred_P, {var_x, const_a}, false);
        // ~P(b,y)
        Literal l2(pred_P, {const_b, var_y}, true);
        auto mgu = Unifier::findMGU(l1, l2, kb);
        Unifier::printSubstitution(mgu.value(), kb);
        Literal l1_new = Unifier::applySubstitutionToLiteral(l1, mgu.value(), kb);
        Literal l2_new = Unifier::applySubstitutionToLiteral(l2, mgu.value(), kb);
        std:: cout << "l1 new " << l1_new.toString(kb) << " l2 new " << l2_new.toString(kb) << std::endl; 
    }

} // namespace LogicSystem