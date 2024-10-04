#ifndef LOGIC_SYSTEM_KNOWLEDGE_BASE_H
#define LOGIC_SYSTEM_KNOWLEDGE_BASE_H

#include <vector>
#include <string>
#include "PredicateTable.h"
#include "VariableTable.h"
#include "ConstantTable.h"
#include "Clause.h"
#include "Fact.h"
#include "SymbolType.h"
#include <map>
#include <optional>

namespace LogicSystem
{
    class KnowledgeBase
    {
    public:
        int addPredicate(const std::string& predicate);
        SymbolId  addVariable(const std::string& variable);
        SymbolId  addConstant(const std::string& constant);
        void addClause(const Clause& clause);
        void addFact(const Fact& fact);

        const std::vector<Clause>& getClauses() const;
        const std::vector<Fact>& getFacts() const;

        std::string getPredicateName(int id) const;
        std::string getSymbolName(const SymbolId& symbolId) const;

        bool isVariable(const SymbolId& symbolId) const;
        bool hasFact(const Fact& fact) const;

        std::optional<int> getPredicateId(const std::string& predicateName) const;
        std::optional<SymbolId> getSymbolId(const std::string& symbolName) const;

        void print() const;

    private:
        PredicateTable predicateTable;
        VariableTable variableTable;
        ConstantTable constantTable;
        std::vector<Clause> clauses;
        std::vector<Fact> facts;

        int nextSkolemFunctionId = 0;
    };
}

#endif // LOGIC_SYSTEM_KNOWLEDGE_BASE_H