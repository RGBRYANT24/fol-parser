#ifndef LOGIC_SYSTEM_CLAUSE_H
#define LOGIC_SYSTEM_CLAUSE_H

#include <vector>
#include <string>
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
        std::vector<Literal> literals;
    private:
        
    };
}

#endif // LOGIC_SYSTEM_CLAUSE_H