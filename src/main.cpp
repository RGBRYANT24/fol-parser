#include <iostream>
#include <memory>
#include "Parser.h"
#include "AllNodes.h"
#include "KnowledgeBase.h"
//#include "CNF.h"
#include "Clause.h"
extern AST::Node* root;

extern "C" int yylex();
extern "C" FILE *yyin;

int main(int argc, char **argv) {
    // 如果提供了输入文件，则从文件读取
    if (argc > 1) {
        FILE *input_file = fopen(argv[1], "r");
        if (!input_file) {
            std::cerr << "无法打开输入文件" << std::endl;
            return 1;
        }
        yyin = input_file;
    }

    // 进行解析
    int parse_result = yyparse();

    if (parse_result == 0) {
        std::cout << "解析成功！" << std::endl;
        
        // 使用解析结果
        if (root) {
            std::cout << "打印 AST：" << std::endl;
            root->print();
        } else {
            std::cout << "AST 为空" << std::endl;
        }
    } else {
        std::cout << "解析失败" << std::endl;
    }
    LogicSystem::CNF* cnf = new LogicSystem::CNF(root);
    cnf -> print();
    LogicSystem::Clause* clause = new LogicSystem::Clause();
    clause -> addLiteral(cnf);
    clause -> print();
    LogicSystem::KnowledgeBase kb;
    kb.addClause(clause);
    kb.print();

    // 如果打开了文件，记得关闭
    if (argc > 1) {
        fclose(yyin);
    }
    //delete root;

    return 0;
}