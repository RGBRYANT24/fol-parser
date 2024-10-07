#include "KnowledgeBaseBuilder.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include "fol.tab.hh"  // Bison 生成的头文件

namespace fs = std::filesystem;

// // 确保这些声明在全局命名空间中
// #ifdef __cplusplus
// extern "C" {
// #endif

extern FILE *yyin;
extern int yylineno;
int yylex();
void yyerror(const char *s);

// #ifdef __cplusplus
// }
// #endif

namespace LogicSystem
{
    KnowledgeBaseBuilder::KnowledgeBaseBuilder() : root(nullptr) {}

    KnowledgeBaseBuilder::~KnowledgeBaseBuilder()
    {
        if (root)
        {
            delete root;
        }
    }

    bool KnowledgeBaseBuilder::parseDirectory(const std::string &input_dir, LogicSystem::KnowledgeBase &kb)
    {
        for (const auto &entry : fs::directory_iterator(input_dir))
        {
            if (entry.path().extension() == ".txt")
            {
                std::string filename = entry.path().string();
                if (readClause(filename, kb))
                {
                    std::cout << "从文件 " << filename << " 添加子句" << std::endl;
                }
                else
                {
                    std::cerr << "处理文件 " << filename << " 时出错" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    bool KnowledgeBaseBuilder::readClause(const std::string &filename, LogicSystem::KnowledgeBase &kb)
    {
        std::ifstream file(filename);
        LogicSystem::Clause clause;
        if (!file.is_open())
        {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
            {
                continue;
            }
            if (parseLiteral(line, kb, clause))
            {
                std::cout << "成功添加字面量 " << line << std::endl;
            }
            else
            {
                std::cerr << "解析 " << line << " 失败" << std::endl;
                return false;
            }
        }
        file.close();
        std::cout << "添加 clause " << std::endl;
        std::cout << clause.toString(kb) << std::endl;
        kb.addClause(clause);
        return true;
    }

    bool KnowledgeBaseBuilder::parseLiteral(const std::string &line, LogicSystem::KnowledgeBase &kb, LogicSystem::Clause &clause)
    {
        if (root)
        {
            delete root;
            root = nullptr;
        }
        yyin = fmemopen((void *)line.c_str(), line.length(), "r");
        if (yyparse() == 0 && root)
        {
            buildKnowledgeBase(root, kb, clause);
            fclose(yyin);
            return true;
        }
        else
        {
            fclose(yyin);
            return false;
        }
    }

    void KnowledgeBaseBuilder::buildKnowledgeBase(AST::Node *node, LogicSystem::KnowledgeBase &kb, LogicSystem::Clause &clause)
    {
        if (node->getType() == AST::Node::PREDICATE)
        {
            handlePredicate(static_cast<AST::PredicateNode *>(node), kb, false, clause);
        }
        else if (node->getType() == AST::Node::NOT)
        {
            AST::UnaryOpNode *notNode = static_cast<AST::UnaryOpNode *>(node);
            if (notNode->child->getType() == AST::Node::PREDICATE)
            {
                handlePredicate(static_cast<AST::PredicateNode *>(notNode->child), kb, true, clause);
            }
        }
    }

    void KnowledgeBaseBuilder::handlePredicate(AST::PredicateNode *node, LogicSystem::KnowledgeBase &kb, bool isNegated, LogicSystem::Clause &clause)
    {
        int predicateId = kb.addPredicate(node->name);
        std::vector<LogicSystem::SymbolId> argumentIds;

        AST::TermListNode *termList = static_cast<AST::TermListNode *>(node->termlists);
        for (AST::Node *arg : termList->arguments)
        {
            if (arg->getType() == AST::Node::VARIABLE)
            {
                argumentIds.push_back(kb.addVariable(arg->name));
            }
            else if (arg->getType() == AST::Node::CONSTANT)
            {
                argumentIds.push_back(kb.addConstant(arg->name));
            }
        }

        LogicSystem::Literal literal(predicateId, argumentIds, isNegated);
        clause.addLiteral(literal);
    }
}
