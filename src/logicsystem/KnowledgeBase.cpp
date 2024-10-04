#include "KnowledgeBase.h"
#include <iostream>
#include <algorithm>

namespace LogicSystem
{
    int KnowledgeBase::addPredicate(const std::string &predicate)
    {
        return predicateTable.insert(predicate);
    }

    SymbolId KnowledgeBase::addVariable(const std::string &variable)
    {
        return {SymbolType::VARIABLE, variableTable.insert(variable)};
    }

    SymbolId KnowledgeBase::addConstant(const std::string &constant)
    {
        return {SymbolType::CONSTANT, constantTable.insert(constant)};
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

    std::string KnowledgeBase::getSymbolName(const SymbolId &symbolId) const
    {
        if (symbolId.type == SymbolType::VARIABLE)
        {
            return variableTable.get(symbolId.id);
        }
        else
        {
            return constantTable.get(symbolId.id);
        }
    }

    std::optional<int> KnowledgeBase::getPredicateId(const std::string &predicateName) const
    {
        return predicateTable.getId(predicateName);
    }

    std::optional<SymbolId> KnowledgeBase::getSymbolId(const std::string &symbolName) const
    {
        // 首先在变量表中查找
        int varId = variableTable.getId(symbolName);
        if (varId != -1)
        {
            return SymbolId{SymbolType::VARIABLE, varId};
        }

        // 然后在常量表中查找
        int constId = constantTable.getId(symbolName);
        if (constId != -1)
        {
            return SymbolId{SymbolType::CONSTANT, constId};
        }

        // 如果都找不到，返回空的 optional
        return std::nullopt;
    }

    bool KnowledgeBase::isVariable(const SymbolId &symbolId) const
    {
        return symbolId.type == SymbolType::VARIABLE;
    }

    bool KnowledgeBase::hasFact(const Fact &fact) const
    {
        return std::find(facts.begin(), facts.end(), fact) != facts.end();
    }

    void KnowledgeBase::print() const
    {
        std::cout << "Knowledge Base:\n";
        std::cout << "Clauses:\n";
        for (const auto &clause : clauses)
        {
            std::cout << "  " << clause.toString(*this) << "\n";
        }
        std::cout << "Facts:\n";
        for (const auto &fact : facts)
        {
            std::cout << "  " << fact.toString(*this) << "\n";
        }
    }
}