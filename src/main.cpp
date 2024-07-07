#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include "Parser.h"
#include "AllNodes.h"
#include "KnowledgeBase.h"
#include "Clause.h"
#include "CNF.h"
#include "Unifier.h"

#include <cassert>
namespace fs = std::filesystem;

extern AST::Node *root;
extern "C" int yylex();
extern "C" FILE *yyin;

// 函数声明
bool readClause(const std::string &filename, LogicSystem::KnowledgeBase& kb);
bool parseLiteral(const std::string& line, LogicSystem::KnowledgeBase& kb);
void buildKnowledgeBase(AST::Node* node, LogicSystem::KnowledgeBase& kb);
void handlePredicate(AST::PredicateNode* node, LogicSystem::KnowledgeBase& kb, bool isNegated);

void testSimpleUnification() {
    LogicSystem::KnowledgeBase kb;
    int predId = kb.addPredicate("P");
    int varX = kb.addVariable("X");
    int varY = kb.addVariable("Y");
    int varZ = kb.addVariable("Z");
    int varZZ = kb.addVariable("ZZ");

    int constA = kb.addConstant("a");

    LogicSystem::Literal lit1(predId, {varX, varZ, constA}, false);
    LogicSystem::Literal lit2(predId, {varY,varZZ, constA}, false);

    auto mgu = LogicSystem::Unifier::findMGU(lit1, lit2, kb);
    assert(mgu.has_value());
    //assert(mgu->size() == 1);
    assert(mgu->at(varX) == varY || mgu->at(varY) == varX);

    // 打印 MGU 内容
    std::cout << "MGU content:" << std::endl;
    for (const auto& [key, value] : *mgu) {
        std::cout << "  " << key << " -> " << value << std::endl;
    }
    std::cout << "Simple unification test passed." << std::endl;
}

void testNonUnifiableLiterals() {
    LogicSystem::KnowledgeBase kb;
    int predP = kb.addPredicate("P");
    int predQ = kb.addPredicate("Q");
    int varX = kb.addVariable("X");
    int constA = kb.addConstant("a");

    LogicSystem::Literal lit1(predP, {varX}, false);
    LogicSystem::Literal lit2(predQ, {constA}, false);

    auto mgu = LogicSystem::Unifier::findMGU(lit1, lit2, kb);
    assert(!mgu.has_value());

    std::cout << "Non-unifiable literals test passed." << std::endl;
}

void testOccursCheck() {
    LogicSystem::KnowledgeBase kb;
    int predId = kb.addPredicate("P");
    int varX = kb.addVariable("X");
    int varY = kb.addVariable("Y");

    LogicSystem::Literal lit1(predId, {varX}, false);
    LogicSystem::Literal lit2(predId, {varY}, false);

    auto mgu = LogicSystem::Unifier::findMGU(lit1, lit2, kb);
    assert(mgu.has_value());

    lit1 = LogicSystem::Literal(predId, {varX}, false);
    lit2 = LogicSystem::Literal(predId, {varX, varY}, false);

    mgu = LogicSystem::Unifier::findMGU(lit1, lit2, kb);
    assert(!mgu.has_value());

    std::cout << "Occurs check test passed." << std::endl;
}


int main()
{
    const std::string input_dir = "../input_files";
    LogicSystem::KnowledgeBase kb;

    for (const auto &entry : fs::directory_iterator(input_dir))
    {
        if (entry.path().extension() == ".txt")
        {
            std::string filename = entry.path().string();
            //LogicSystem::Clause *clause = parseFileToClause(filename);
            bool addClause = readClause(filename, kb);

            if (addClause)
            {
                std::cout << "从文件 " << filename << " 添加子句" <<std::endl;
            }
            else
            {
                std::cerr << "处理文件 " << filename << " 时出错" << std::endl;
            }
        }
    }

    std::cout << "\n最终知识库：" << std::endl;
    kb.print();

    testSimpleUnification();
    testNonUnifiableLiterals();
    testOccursCheck();

    std::cout << "All tests passed!" << std::endl;
    //unifierTest();

}

bool readClause(const std::string &filename, LogicSystem::KnowledgeBase& kb)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
         // 跳过空行
        if (line.empty())
        {
            continue;
        }
        if(parseLiteral(line, kb))
        {
            std::cout << "成功添加字面量 " << line << std::endl; // 调试输出
            continue;
        }
        else
        {
            std::cerr << "解析 " << line << " 失败" << std::endl;
            return false;
        }
    }
    file.close();
    return true;
}

bool parseLiteral(const std::string& line, LogicSystem::KnowledgeBase& kb)
{
    // 重置解析状态
    if (root)
    {
        delete root;
        root = nullptr;
    }
    // 设置输入为当前literal
    yyin = fmemopen((void *)line.c_str(), line.length(), "r");
    if (yyparse() == 0 && root)
    {
        buildKnowledgeBase(root, kb);// add literal to kb
        //delete root;
        fclose(yyin);
        return true;
    }
    else
    {
        fclose(yyin);
        return false;
    }
}

void buildKnowledgeBase(AST::Node* node, LogicSystem::KnowledgeBase& kb) {
    if (node->getType() == AST::Node::PREDICATE) {
        handlePredicate(static_cast<AST::PredicateNode*>(node), kb, false);
    } else if (node->getType() == AST::Node::NOT) {
        AST::UnaryOpNode* notNode = static_cast<AST::UnaryOpNode*>(node);
        if (notNode->child->getType() == AST::Node::PREDICATE) {
            handlePredicate(static_cast<AST::PredicateNode*>(notNode->child), kb, true);
        }
    }
}

void handlePredicate(AST::PredicateNode* node, LogicSystem::KnowledgeBase& kb, bool isNegated) {
    int predicateId = kb.addPredicate(node->name);
    std::vector<int> argumentIds;
    
    AST::TermListNode* termList = static_cast<AST::TermListNode*>(node->termlists);
    for (AST::Node* arg : termList->arguments) {
        if (arg->getType() == AST::Node::VARIABLE) {
            argumentIds.push_back(kb.addVariable(arg->name));
        } else if (arg->getType() == AST::Node::CONSTANT) {
            argumentIds.push_back(kb.addConstant(arg->name));
        }
    }
    //delete node; //AST 的析构函数可能还是有问题
    
    LogicSystem::Literal literal(predicateId, argumentIds, isNegated);
    LogicSystem::Clause clause;
    clause.addLiteral(literal);
    kb.addClause(clause);
}