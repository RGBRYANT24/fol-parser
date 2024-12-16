#include <gtest/gtest.h>
#include "SLITree.h"
#include "KnowledgeBase.h"
#include "VariableRenamer.h"

using namespace LogicSystem;

class VariableRenamerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set up predicates
        pred_P = kb.addPredicate("P"); // binary predicate
        pred_Q = kb.addPredicate("Q"); // unary predicate

        // Set up constants
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");

        // Set up variables
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
        var_z = kb.addVariable("z");
    }

    KnowledgeBase kb;
    int pred_P;
    int pred_Q;
    SymbolId const_a;
    SymbolId const_b;
    SymbolId var_x;
    SymbolId var_y;
    SymbolId var_z;
};

TEST_F(VariableRenamerTest, CollectVariables)
{
    // Test collectClauseVariables
    Clause clause;
    // Add P(x,y)
    clause.addLiteral(Literal(pred_P, {var_x, var_y}, false));
    // Add Q(z)
    clause.addLiteral(Literal(pred_Q, {var_z}, false));
    // Add P(x,a)
    clause.addLiteral(Literal(pred_P, {var_x, const_a}, false));

    auto clause_vars = VariableRenamer::collectClauseVariables(clause, kb);

    // Should contain x, y, z
    EXPECT_EQ(clause_vars.size(), 3);
    EXPECT_TRUE(clause_vars.find(var_x) != clause_vars.end());
    EXPECT_TRUE(clause_vars.find(var_y) != clause_vars.end());
    EXPECT_TRUE(clause_vars.find(var_z) != clause_vars.end());

    // Test collectTreeVariables
    SLITree tree(kb);

    // Add nodes to the tree
    Clause c1;
    c1.addLiteral(Literal(pred_P, {var_x, var_y}, false)); // P(x,y)
    tree.add_node(c1, Literal(), false, tree.getRoot());

    Clause c2;
    c2.addLiteral(Literal(pred_Q, {var_z}, false)); // Q(z)
    tree.add_node(c2, Literal(), false, tree.getRoot());

    auto tree_vars = VariableRenamer::collectTreeVariables(tree, kb);

    // Should contain x, y, z
    EXPECT_EQ(tree_vars.size(), 3);
    EXPECT_TRUE(tree_vars.find(var_x) != tree_vars.end());
    EXPECT_TRUE(tree_vars.find(var_y) != tree_vars.end());
    EXPECT_TRUE(tree_vars.find(var_z) != tree_vars.end());

    // Test with constants (should not be collected)
    Clause c3;
    c3.addLiteral(Literal(pred_P, {const_a, const_b}, false)); // P(a,b)
    tree.add_node(c3, Literal(), false, tree.getRoot());

    auto tree_vars_with_constants = VariableRenamer::collectTreeVariables(tree, kb);
    EXPECT_EQ(tree_vars_with_constants.size(), 3); // Should still be 3, constants not included
}

TEST_F(VariableRenamerTest, EmptyCollections)
{
    // Test empty clause
    Clause empty_clause;
    auto empty_clause_vars = VariableRenamer::collectClauseVariables(empty_clause, kb);
    EXPECT_TRUE(empty_clause_vars.empty());

    // Test tree with only constants
    SLITree tree(kb);
    Clause const_clause;
    const_clause.addLiteral(Literal(pred_P, {const_a, const_b}, false)); // P(a,b)
    tree.add_node(const_clause, Literal(), false, tree.getRoot());

    auto tree_const_vars = VariableRenamer::collectTreeVariables(tree, kb);
    EXPECT_TRUE(tree_const_vars.empty());
}

TEST_F(VariableRenamerTest, RenameClauseTest)
{
    // 场景1：基本变量重命名
    SLITree tree(kb);
    Clause treeClause;
    treeClause.addLiteral(Literal(pred_P, {var_x, var_y}, false)); // P(x,y)
    tree.add_node(treeClause, Literal(), false, tree.getRoot());

    Clause inputClause;
    inputClause.addLiteral(Literal(pred_P, {var_x, var_y}, false)); // P(x,y)

    Clause renamedClause = VariableRenamer::renameClauseVariables(inputClause, tree, kb);

    // 检查重命名后的变量是否与原变量不同
    std::cout << "After Rename Clause " << renamedClause.toString(kb) << std::endl;
    auto literals = renamedClause.getLiterals();
    ASSERT_EQ(literals.size(), 1);
    auto args = literals[0].getArgumentIds();
    EXPECT_NE(args[0], var_x);
    EXPECT_NE(args[1], var_y);
    EXPECT_TRUE(kb.isVariable(args[0]));
    EXPECT_TRUE(kb.isVariable(args[1]));

    // 场景2：包含常量的子句重命名
    Clause mixedClause;
    mixedClause.addLiteral(Literal(pred_P, {var_x, const_a}, false)); // P(x,a)

    Clause renamedMixedClause = VariableRenamer::renameClauseVariables(mixedClause, tree, kb);

    std::cout << "After Rename Clause " << renamedMixedClause.toString(kb) << std::endl;
    auto mixedLiterals = renamedMixedClause.getLiterals();
    ASSERT_EQ(mixedLiterals.size(), 1);
    auto mixedArgs = mixedLiterals[0].getArgumentIds();
    EXPECT_NE(mixedArgs[0], var_x);
    EXPECT_EQ(mixedArgs[1], const_a); // 常量应保持不变
    EXPECT_TRUE(kb.isVariable(mixedArgs[0]));
    EXPECT_FALSE(kb.isVariable(mixedArgs[1]));

    // 场景3：变量名冲突处理
    SLITree conflictTree(kb);
    Clause treeClause2;
    SymbolId var_x1 = kb.addVariable("x1");
    treeClause2.addLiteral(Literal(pred_P, {var_x1, var_y}, false)); // P(x1,y)
    conflictTree.add_node(treeClause2, Literal(), false, conflictTree.getRoot());

    Clause inputClause2;
    inputClause2.addLiteral(Literal(pred_P, {var_x, var_y}, false)); // P(x,y)

    Clause renamedConflictClause = VariableRenamer::renameClauseVariables(inputClause2, conflictTree, kb);
    //换名的结果实际上应该是P(x, y1) 因为第一个x没有重名，所以其实不用换名
    std::cout << "After Rename Clause " << renamedConflictClause.toString(kb) << std::endl;
    auto conflictLiterals = renamedConflictClause.getLiterals();
    ASSERT_EQ(conflictLiterals.size(), 1);
    auto conflictArgs = conflictLiterals[0].getArgumentIds();
    EXPECT_EQ(conflictArgs[0], var_x);//
    EXPECT_NE(conflictArgs[0], var_x1);
    EXPECT_NE(conflictArgs[1], var_y);
    
    // 验证新生成的变量名
    std::string newXName = kb.getSymbolName(conflictArgs[0]);
    std::string newYName = kb.getSymbolName(conflictArgs[1]);
    EXPECT_TRUE(newXName.find("x") == 0); // 应该以 'x' 开头
    EXPECT_TRUE(newYName.find("y") == 0); // 应该以 'y' 开头

     // 场景4：空子句重命名
    Clause emptyClause;
    Clause renamedEmptyClause = VariableRenamer::renameClauseVariables(emptyClause, tree, kb);
    EXPECT_TRUE(renamedEmptyClause.getLiterals().empty());

     // 场景5：多文字子句重命名
    Clause multiLitClause;
    multiLitClause.addLiteral(Literal(pred_P, {var_x, const_a}, false)); // P(x,y)
    multiLitClause.addLiteral(Literal(pred_Q, {var_x}, false));        // Q(x)
    
    Clause renamedMultiLitClause = VariableRenamer::renameClauseVariables(multiLitClause, tree, kb);
    std::cout << "After Rename Clause " << renamedMultiLitClause.toString(kb) << std::endl;
    auto multiLiterals = renamedMultiLitClause.getLiterals();
    ASSERT_EQ(multiLiterals.size(), 2);
    
    // 确保同一个变量在不同文字中被一致地重命名
    SymbolId renamedX1 = multiLiterals[0].getArgumentIds()[0];
    SymbolId renamedX2 = multiLiterals[1].getArgumentIds()[0];
    EXPECT_EQ(renamedX1, renamedX2);
}
