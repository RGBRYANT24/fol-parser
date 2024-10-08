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
        Literal(int predId, const std::vector<SymbolId> &argIds, bool negated);

        std::string toString(const KnowledgeBase &kb) const;

        int getPredicateId() const;
        const std::vector<SymbolId> &getArgumentIds() const;
        bool isNegated() const;
        // 添加 == 运算符
        bool operator==(const Literal &other) const;
        // 添加 != 运算符
        bool operator!=(const Literal &other) const;

        size_t hash() const
        {
            size_t h = std::hash<int>{}(predicateId);
            for (const auto &arg : argumentIds)
            {
                h ^= std::hash<SymbolId>{}(arg) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            h ^= std::hash<bool>{}(negated);
            return h;
        }

    private:
        int predicateId;
        std::vector<SymbolId> argumentIds;
        bool negated;
    };
}

namespace std
{
    template <>
    struct hash<LogicSystem::Literal>
    {
        size_t operator()(const LogicSystem::Literal &lit) const
        {
            return lit.hash();
        }
    };
}
#endif // LOGIC_SYSTEM_LITERAL_H