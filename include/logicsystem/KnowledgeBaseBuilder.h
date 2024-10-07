#ifndef KNOWLEDGE_BASE_BUILDER_H
#define KNOWLEDGE_BASE_BUILDER_H

#include <string>
#include <vector>
#include "KnowledgeBase.h"
#include "AllNodes.h"
// #include "LogicSystem.h"
// #include "AST.h"

namespace LogicSystem
{
    class KnowledgeBaseBuilder
    {
    public:
        KnowledgeBaseBuilder();
        ~KnowledgeBaseBuilder();

        bool parseDirectory(const std::string &input_dir, LogicSystem::KnowledgeBase &kb);
        bool readClause(const std::string &filename, LogicSystem::KnowledgeBase &kb);

    private:
        AST::Node *root;

        bool parseLiteral(const std::string &line, LogicSystem::KnowledgeBase &kb, LogicSystem::Clause &clause);
        void buildKnowledgeBase(AST::Node *node, LogicSystem::KnowledgeBase &kb, LogicSystem::Clause &clause);
        void handlePredicate(AST::PredicateNode *node, LogicSystem::KnowledgeBase &kb, bool isNegated, LogicSystem::Clause &clause);
    };
}

#endif // KNOWLEDGE_BASE_BUILDER_H