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

        // 添加默认构造函数，创建一个特殊的"空"文字
        Literal() : predicateId(-1), argumentIds(), negated(false) {} // 使用 -1 表示特殊的空谓词

        // 添加一个方法来检查是否是空文字
        bool isEmpty() const { return predicateId == -1; }

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