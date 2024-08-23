#ifndef LOGIC_SYSTEM_CLAUSE_H
#define LOGIC_SYSTEM_CLAUSE_H

#include <vector>
#include <string>
#include <unordered_map>
#include "Literal.h"

namespace LogicSystem
{
    class KnowledgeBase;

    class Clause
    {
    public:
        void addLiteral(const Literal& lit);
        const std::vector<Literal>& getLiterals() const;
        bool isEmpty() const;
        std::string toString(const KnowledgeBase& kb) const;
        
    private:
        std::unordered_map<int, int> literalMap;//PredicateId -> 出现次数
        std::vector<Literal> literals;

        bool hasOppositeLiteral(const Literal& lit) const;
    };
}

#endif // LOGIC_SYSTEM_CLAUSE_H