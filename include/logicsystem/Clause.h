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
        void addLiteral(const Literal &lit);
        const std::vector<Literal> &getLiterals() const;
        bool isEmpty() const;
        std::string toString(const KnowledgeBase &kb) const;
        bool isTautology() const; // 检查是否为重言式
        // 添加检测函数，检查是否包含 E(x, x) 这样的文字
        bool containsSelfLoop(const KnowledgeBase &kb) const;
        // 默认构造函数
        Clause() = default;

        // 复制构造函数
        Clause(const Clause &other);

        // 赋值运算符
        Clause &operator=(const Clause &other);

        // 移动构造函数
        Clause(Clause &&other) noexcept;

        // 移动赋值运算符
        Clause &operator=(Clause &&other) noexcept;

        //hash计算函数
        size_t hash() const;
        bool operator==(const Clause &other) const;

    private:
        std::unordered_map<int, int> literalMap; // PredicateId -> 下标
        std::vector<Literal> literals;
        bool isTautologyFlag = false; // 标记是否为重言式

        bool hasOppositeLiteral(const Literal &lit) const;

        mutable size_t hashValue;
        mutable bool hashComputed = false;
    };
}

#endif // LOGIC_SYSTEM_CLAUSE_H