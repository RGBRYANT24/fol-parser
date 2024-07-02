#ifndef LOGIC_SYSTEM_KNOWLEDGE_BASE_H
#define LOGIC_SYSTEM_KNOWLEDGE_BASE_H

#include <vector>
#include <string>
#include "PredicateTable.h"
#include "VariableTable.h"
#include "ConstantTable.h"
#include "Clause.h"
#include "Fact.h"
#include <map>

namespace LogicSystem
{
    class KnowledgeBase
    {
    public:
        int addPredicate(const std::string& predicate);
        int addVariable(const std::string& variable);
        int addConstant(const std::string& constant);
        void addClause(const Clause& clause);
        void addFact(const Fact& fact);

        const std::vector<Clause>& getClauses() const;
        const std::vector<Fact>& getFacts() const;

        std::string getPredicateName(int id) const;
        std::string getVariableName(int id) const;
        std::string getConstantName(int id) const;

        bool isVariable(int id) const;
        bool hasFact(const Fact& fact) const;

        void standardizeVariables();//变量标准化，避免重名
        void setStandardizedVariableName(int id, const std::string& name);
        std::string getOriginalVariableName(int id) const;
        void print() const;

    private:
        PredicateTable predicateTable;
        VariableTable variableTable;
        ConstantTable constantTable;
        std::vector<Clause> clauses;
        std::vector<Fact> facts;
        std::map<int, std::string> standardizedVariableNames; //存储标准化之后的变量名

        
        void standardizeClause(Clause& clause);
        int nextVariableId = 0;
        int nextSkolemFunctionId = 0;
    };
}

#endif // LOGIC_SYSTEM_KNOWLEDGE_BASE_H