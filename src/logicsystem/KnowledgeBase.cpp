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

    void KnowledgeBase::setStandardizedVariableName(int id, const std::string &name)
    {
        standardizedVariableNames[id] = name;
    }

    void KnowledgeBase::standardizeVariables()
    {
        int nextStandardId = 0;
        std::map<int, int> variableMap;

        for (Clause &clause : clauses)
        {
            for (Literal &literal : clause.literals)
            {
                for (int &argId : literal.argumentIds)
                {
                    if (isVariable(argId))
                    {
                        if (variableMap.find(argId) == variableMap.end())
                        {
                            int newId = nextVariableId++;
                            variableMap[argId] = newId;
                            // 设置新的标准化变量名
                            setStandardizedVariableName(newId, "X" + std::to_string(nextStandardId++));
                        }
                        argId = variableMap[argId];
                    }
                }
            }
        }
    }

    void KnowledgeBase::standardizeClause(Clause &clause)
    {
        std::map<int, int> variableMap;
        std::vector<Literal> literals = clause.getLiterals();
        for (Literal &literal : literals)
        {
            std::vector<int> arguments = literal.getArgumentIds();
            for (int &argId : arguments)
            {
                if (isVariable(argId))
                {
                    if (variableMap.find(argId) == variableMap.end())
                    {
                        variableMap[argId] = nextVariableId++;
                    }
                    argId = variableMap[argId];
                }
            }
        }
    }

    void KnowledgeBase::print() const
    {
        for (const Clause &clause : clauses)
        {
            for (const Literal &literal : clause.literals)
            {
                if (literal.negated)
                {
                    std::cout << "¬";
                }
                std::cout << getPredicateName(literal.predicateId) << "(";
                for (size_t i = 0; i < literal.argumentIds.size(); ++i)
                {
                    int argId = literal.argumentIds[i];
                    if (isVariable(argId))
                    {
                        std::cout << getVariableName(argId);
                    }
                    else
                    {
                        std::cout << getConstantName(argId);
                    }
                    if (i < literal.argumentIds.size() - 1)
                    {
                        std::cout << ", ";
                    }
                }
                std::cout << ") ";
            }
            std::cout << std::endl;
        }
    }
}