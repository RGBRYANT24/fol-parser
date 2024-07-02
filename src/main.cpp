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
#include "Resolver.h"

namespace fs = std::filesystem;

extern AST::Node *root;
extern "C" int yylex();
extern "C" FILE *yyin;

// 函数声明
LogicSystem::Clause *parseFileToClause(const std::string &filename);
LogicSystem::CNF *parseLiteral(const std::string &literal);

int main()
{
    const std::string input_dir = "../input_files";
    LogicSystem::KnowledgeBase kb;

    for (const auto &entry : fs::directory_iterator(input_dir))
    {
        if (entry.path().extension() == ".txt")
        {
            std::string filename = entry.path().string();
            LogicSystem::Clause *clause = parseFileToClause(filename);

            if (clause)
            {
                kb.addClause(clause);
                std::cout << "从文件 " << filename << " 添加了子句：";
                clause->print();
                std::cout << std::endl;
            }
            else
            {
                std::cerr << "处理文件 " << filename << " 时出错" << std::endl;
            }
        }
    }

    std::cout << "\n最终知识库：" << std::endl;
    kb.print();

    LogicSystem::Resolver resolver;
    bool isSatisfiable = resolver.isSatisfiable(kb);

    if (isSatisfiable)
    {
        std::cout << "The knowledge base is satisfiable." << std::endl;
    }
    else
    {
        std::cout << "The knowledge base is unsatisfiable." << std::endl;
    }
    return 0;
}

LogicSystem::Clause *parseFileToClause(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return nullptr;
    }

    LogicSystem::Clause *clause = new LogicSystem::Clause();
    std::string line;

    while (std::getline(file, line))
    {
        // 跳过空行
        if (line.empty())
        {
            continue;
        }

        LogicSystem::CNF *cnf = parseLiteral(line);
        if (cnf)
        {
            clause->addLiteral(cnf);
            std::cout << "成功添加字面量"; // 调试输出
            cnf->print();
        }
        else
        {
            std::cerr << "解析 " << line << " 失败" << std::endl;
        }
    }

    file.close();
    return clause;
}

LogicSystem::CNF *parseLiteral(const std::string &literal)
{
    // 重置解析状态
    if (root)
    {
        delete root;
        root = nullptr;
    }

    // 设置输入为当前literal
    yyin = fmemopen((void *)literal.c_str(), literal.length(), "r");

    if (yyparse() == 0 && root)
    {
        LogicSystem::CNF *cnf = new LogicSystem::CNF(root);
        fclose(yyin);
        return cnf;
    }
    else
    {
        fclose(yyin);
        return nullptr;
    }
}