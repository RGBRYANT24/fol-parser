#ifndef LOGIC_SYSTEM_LITERAL_H
#define LOGIC_SYSTEM_LITERAL_H

#include <vector>
#include <string>
#include "SymbolType.h"

namespace LogicSystem
{
    class KnowledgeBase;

    class Literal
    {
    public:
        Literal(int predId, const std::vector<SymbolId>& argIds, bool negated);

        std::string toString(const KnowledgeBase& kb) const;

        int getPredicateId() const;
        const std::vector<SymbolId>& getArgumentIds() const;
        bool isNegated() const;
        // 添加 == 运算符
        bool operator==(const Literal& other) const;
        // 添加 != 运算符
        bool operator!=(const Literal& other) const;


        
    private:
        int predicateId;
        std::vector<SymbolId> argumentIds;
        bool negated;
        
    };
}

#endif // LOGIC_SYSTEM_LITERAL_H