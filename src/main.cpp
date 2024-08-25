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
#include <cassert>
namespace fs = std::filesystem;

extern AST::Node *root;
extern "C" int yylex();
extern "C" FILE *yyin;

// 函数声明
bool readClause(const std::string &filename, LogicSystem::KnowledgeBase& kb);
bool parseLiteral(const std::string& line, LogicSystem::KnowledgeBase& kb, LogicSystem::Clause& clause);
void buildKnowledgeBase(AST::Node* node, LogicSystem::KnowledgeBase& kb, LogicSystem::Clause& clause);
void handlePredicate(AST::PredicateNode* node, LogicSystem::KnowledgeBase& kb, bool isNegated, LogicSystem::Clause& clause);
bool resolutionTest();
void addClauseTest();

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

   resolutionTest();
   //addClauseTest();
   return 0;
}

bool readClause(const std::string &filename, LogicSystem::KnowledgeBase& kb)
{
    std::ifstream file(filename);
    LogicSystem::Clause clause; // 每个文件应该保存在一个clause中
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
        if(parseLiteral(line, kb, clause))
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
    std::cout << "添加 clause " << std::endl;
    std::cout << clause.toString(kb) << std::endl;
    kb.addClause(clause);
    return true;
}

bool parseLiteral(const std::string& line, LogicSystem::KnowledgeBase& kb, LogicSystem::Clause& clause)
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
        buildKnowledgeBase(root, kb, clause);// add literal to kb
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

void buildKnowledgeBase(AST::Node* node, LogicSystem::KnowledgeBase& kb, LogicSystem::Clause& clause) {
    if (node->getType() == AST::Node::PREDICATE) {
        handlePredicate(static_cast<AST::PredicateNode*>(node), kb, false, clause);
    } else if (node->getType() == AST::Node::NOT) {
        AST::UnaryOpNode* notNode = static_cast<AST::UnaryOpNode*>(node);
        if (notNode->child->getType() == AST::Node::PREDICATE) {
            handlePredicate(static_cast<AST::PredicateNode*>(notNode->child), kb, true, clause);
        }
    }
}

void handlePredicate(AST::PredicateNode* node, LogicSystem::KnowledgeBase& kb, bool isNegated, LogicSystem::Clause& clause) {
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
    //LogicSystem::Clause clause;
    clause.addLiteral(literal);
    //kb.addClause(clause);
}

bool resolutionTest()
{
    LogicSystem::KnowledgeBase kb;

    // 添加谓词
    int dogId = kb.addPredicate("Dog");
    int animalId = kb.addPredicate("Animal");
    int breathesId = kb.addPredicate("Breathes");
    int ownsId = kb.addPredicate("Owns");

    // 添加常量
    int xiaomingId = kb.addVariable("Xiaoming");

    // 添加变量
    int xId = kb.addVariable("X");
    int yId = kb.addVariable("Y");
    //std::cout <<"xId " << xId << " yId " << yId << std::endl;
    // 1. 所有的狗都是动物: ∀X(Dog(X) → Animal(X))
    // 转换为子句形式: ¬Dog(X) ∨ Animal(X)
    LogicSystem::Clause clause1;
    clause1.addLiteral(LogicSystem::Literal(dogId, {xId}, true));  // ¬Dog(X)
    clause1.addLiteral(LogicSystem::Literal(animalId, {xId}, false));  // Animal(X)
    kb.addClause(clause1);

    // 2. 所有的动物都会呼吸: ∀Y(Animal(Y) → Breathes(Y))
    // 转换为子句形式: ¬Animal(Y) ∨ Breathes(Y)
    LogicSystem::Clause clause2;
    clause2.addLiteral(LogicSystem::Literal(animalId, {yId}, true));  // ¬Animal(Y)
    clause2.addLiteral(LogicSystem::Literal(breathesId, {yId}, false));  // Breathes(Y)
    kb.addClause(clause2);

    // 3. 小明养了一只狗: Dog(Xiaoming's_dog) ∧ Owns(Xiaoming, Xiaoming's_dog)
    // 我们将这拆分为两个事实
    LogicSystem::Clause factClause1, factClause2;
    int xiaomingDogId = kb.addVariable("Xiaoming's_dog");
    //std::cout << "xiaomingDogId "<< xiaomingDogId << std::endl;
    factClause1.addLiteral(LogicSystem::Literal(dogId, {xiaomingDogId}, false));
    factClause2.addLiteral(LogicSystem::Literal(ownsId, {xiaomingId, xiaomingDogId},false));
    //factClause.addLiteral(LogicSystem::Literal(ownsId, {xiaomingId, xiaomingDogId},true));
    kb.addClause(factClause1);
    kb.addClause(factClause2);
    kb.print();
    //std::cout << "factClause " << factClause.toString(kb) << std::endl;
    /*LogicSystem::Fact fact1(dogId, {kb.addConstant("Xiaoming's_dog")});
    kb.addFact(fact1);*/

    /*LogicSystem::Fact fact2(ownsId, {xiaomingId, kb.addConstant("Xiaoming's_dog")});
    kb.addFact(fact2);*/

    // 创建目标子句：证明小明养的动物会呼吸
    // 目标: ∃Z(Owns(Xiaoming, Z) ∧ Animal(Z) ∧ Breathes(Z))
    // 否定后: ∀Z(¬Owns(Xiaoming, Z) ∨ ¬Animal(Z) ∨ ¬Breathes(Z))
    int zId = kb.addVariable("Z");
    LogicSystem::Clause goal;
    goal.addLiteral(LogicSystem::Literal(ownsId, {xiaomingId, zId}, true));  // ¬Owns(Xiaoming, Z)
    goal.addLiteral(LogicSystem::Literal(animalId, {zId}, true));  // ¬Animal(Z)
    goal.addLiteral(LogicSystem::Literal(breathesId, {zId}, true));  // ¬Breathes(Z)

    /*kb.addClause(goal);
    std::cout << "print kb" << std::endl;
    kb.print();
    std::optional<LogicSystem::Clause> result = LogicSystem::Resolution::testResolve(clause1, factClause, 0, 0, kb);

    if (result) {
        std::cout << "Resolution result: " << result->toString(kb) << std::endl;
    } else {
        std::cout << "Resolution failed." << std::endl;
    }

    return 1;*/



    // 尝试证明
    //bool proved = LogicSystem::Resolution::prove(kb, goal);
    //bool proved = LogicSystem::Resolution::proveDFS(kb, goal);
    bool proved = LogicSystem::Resolution::proveBFS(kb, goal);

    if (proved) {
        std::cout << "Goal proved: Xiaoming owns an animal that breathes!" << std::endl;
    } else {
        std::cout << "Unable to prove the goal." << std::endl;
    }
    return proved;
}

void addClauseTest()
{
    LogicSystem::KnowledgeBase kb;

    // 添加谓词
    int dogId = kb.addPredicate("Dog");
    int animalId = kb.addPredicate("Animal");
    int breathesId = kb.addPredicate("Breathes");
    int ownsId = kb.addPredicate("Owns");

    // 添加常量
    int xiaomingId = kb.addVariable("Xiaoming");

    // 添加变量
    int xId = kb.addVariable("X");
    int yId = kb.addVariable("Y");


    LogicSystem::Clause clause1;
    clause1.addLiteral(LogicSystem::Literal(dogId, {xId}, true));  // ¬Dog(X)
    clause1.addLiteral(LogicSystem::Literal(animalId, {xId}, false));  // Animal(X)
    std::cout << "before " << clause1.toString(kb) << std::endl;
    clause1.addLiteral(LogicSystem::Literal(animalId, {xId}, false));  // Animal(X)
    clause1.addLiteral(LogicSystem::Literal(breathesId, {yId}, true));  // Animal(Y)
    std::cout << "after " << clause1.toString(kb) << std::endl;


}   