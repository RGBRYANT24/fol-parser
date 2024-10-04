#ifndef LOGIC_SYSTEM_FACT_H
#define LOGIC_SYSTEM_FACT_H

#include <vector>
#include <string>
#include "SymbolType.h"

namespace LogicSystem
{
    class KnowledgeBase;

    class Fact
    {
    public:
        Fact(int predId, const std::vector<SymbolId>& argIds);

        int getPredicateId() const;
        const std::vector<SymbolId>& getArgumentIds() const;

        bool operator==(const Fact& other) const;

        std::string toString(const KnowledgeBase& kb) const;

    private:
        int predicateId;
        std::vector<SymbolId> argumentIds;
    };
}

#endif // LOGIC_SYSTEM_FACT_H