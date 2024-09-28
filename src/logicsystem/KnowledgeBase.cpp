#include "KnowledgeBase.h"
#include <iostream>
#include <algorithm>

namespace LogicSystem
{
    int KnowledgeBase::addPredicate(const std::string &predicate)
    {
        return predicateTable.insert(predicate);
    }

    int KnowledgeBase::addVariable(const std::string &variable)
    {
        return variableTable.insert(variable);
    }

    int KnowledgeBase::addConstant(const std::string &constant)
    {
        return constantTable.insert(constant);
    }

    void KnowledgeBase::addClause(const Clause &clause)
    {
        clauses.push_back(clause);
    }

    void KnowledgeBase::addFact(const Fact &fact)
    {
        if (!hasFact(fact))
        {
            facts.push_back(fact);
        }
    }

    const std::vector<Clause> &KnowledgeBase::getClauses() const
    {
        return clauses;
    }

    const std::vector<Fact> &KnowledgeBase::getFacts() const
    {
        return facts;
    }

    std::string KnowledgeBase::getPredicateName(int id) const
    {
        return predicateTable.get(id);
    }

    std::string KnowledgeBase::getVariableName(int id) const
    {
        return variableTable.get(id);
    }

    std::string KnowledgeBase::getConstantName(int id) const
    {
        return constantTable.get(id);
    }

    bool KnowledgeBase::isVariable(int id) const
    {
        return id < variableTable.size();
    }

    bool KnowledgeBase::hasFact(const Fact &fact) const
    {
        return std::find(facts.begin(), facts.end(), fact) != facts.end();
    }

    void KnowledgeBase::print() const {
        std::cout << "Knowledge Base:\n";
        std::cout << "Clauses:\n";
        for (const auto& clause : clauses) {
            std::cout << "  " << clause.toString(*this) << "\n";
        }
        std::cout << "Facts:\n";
        for (const auto& fact : facts) {
            std::cout << "  " << fact.toString(*this) << "\n";
        }
    }
}