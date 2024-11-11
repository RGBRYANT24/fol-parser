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
#include "Resolution.h"
#include "SymbolType.h"
#include <cassert>
namespace fs = std::filesystem;


int main()
{
    /*const std::string input_dir = "../input_files";
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
    kb.print();*/

    /*LogicSystem::Clause goal;
    int xiaomingID = kb.addConstant("xiaoming");
    int PredicateID = kb.addPredicate("R");
    goal.addLiteral(LogicSystem::Literal(PredicateID,{xiaomingID}, false));

    bool proved = LogicSystem::Resolution::prove(kb, goal);
    if (proved) {
        std::cout << "Goal proved!" << std::endl;
    } else {
        std::cout << "Unable to prove the goal." << std::endl;
    }*/

   //resolutionTest();
   //isTautologyTest();
   //addClauseTest();
   return 0;
}
