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
        bool isTautology() const; // 检查是否为重言式
        // 默认构造函数
        Clause() = default;

        // 复制构造函数
        Clause(const Clause& other);

        // 赋值运算符
        Clause& operator=(const Clause& other);

        // 移动构造函数
        Clause(Clause&& other) noexcept;

        // 移动赋值运算符
        Clause& operator=(Clause&& other) noexcept;
        
    private:
        std::unordered_map<int, int> literalMap;//PredicateId -> 出现次数
        std::vector<Literal> literals;
        bool isTautologyFlag; // 标记是否为重言式

        bool hasOppositeLiteral(const Literal& lit) const;
    };
}

#endif // LOGIC_SYSTEM_CLAUSE_H