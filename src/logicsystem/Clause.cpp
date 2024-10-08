#include "Clause.h"
#include "KnowledgeBase.h"
#include <iostream>
#include <unordered_set>

namespace LogicSystem
{
    void Clause::addLiteral(const Literal &lit)
    {
        int predicateId = lit.getPredicateId();
        // std::cout << "PredicatedId " << predicateId << std::endl;

        // TOOD: 存在互补的谓词，也要判断是不是都是变量，都是变量才是重言式。而一个变量一个常量就不是
        if (this->hasOppositeLiteral(lit)) // 如果存在互补的文字,不添加,直接删除
        {
            auto it = this->literalMap.find(predicateId);
            if (it != this->literalMap.end())
            {
                // 发现重言式
                isTautologyFlag = true;
                literals.push_back(lit);
                // this->literalMap[predicateId] = this->literals.size() - 1;
                return;
            }
        }
        else if (this->literalMap.find(predicateId) != this->literalMap.end()) // 如果已经有相同
        {
            // 如果已经有相同谓词的文字项，检查是否完全相同
            size_t existingLiteralIndex = this->literalMap[predicateId];
            const Literal &existingLiteral = this->literals[existingLiteralIndex];
            if (lit == existingLiteral)
            {
                // std::cout << "Literal is identical to existing one, skipping addition" << std::endl;
                return;
            }
        }

        // 如果不存在相同的文字项，或者存在但不完全相同，则添加新的文字项
        literals.push_back(lit);
        this->literalMap[predicateId] = this->literals.size() - 1;
    }

    const std::vector<Literal> &Clause::getLiterals() const
    {
        return literals;
    }

    bool Clause::isEmpty() const
    {
        return literals.empty();
    }

    std::string Clause::toString(const KnowledgeBase &kb) const
    {
        std::string result;
        if (isTautologyFlag)
        {
            result += "T (Tautology): ";
        }
        for (size_t i = 0; i < literals.size(); ++i)
        {
            result += literals[i].toString(kb);
            if (i < literals.size() - 1)
            {
                result += " ∨ ";
            }
        }
        return result;
    }

    bool Clause::hasOppositeLiteral(const Literal &lit) const
    {
        int nameId = lit.getPredicateId();
        // 如果出现相同谓词并且互补
        // TODO: 判断出现完全相同参数/变量
        if (this->literalMap.find(nameId) != this->literalMap.end() && this->literals[this->literalMap.at(nameId)].isNegated() != lit.isNegated())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Clause::isTautology() const
    {
        return isTautologyFlag;
    }

    bool Clause::containsSelfLoop(const KnowledgeBase &kb) const
    {
        for (const auto &lit : literals)
        {
            // 检查谓词是否为 E
            if (kb.getPredicateName(lit.getPredicateId()) == "E")
            {
                const auto &args = lit.getArgumentIds();
                // 检查参数是否相同
                if (args.size() == 2 && args[0] == args[1])
                {
                    return true;
                }
            }
        }
        return false;
    }

    // 复制构造函数
    Clause::Clause(const Clause &other)
        : literalMap(other.literalMap), literals(other.literals)
    {
        // 由于 std::unordered_map 和 std::vector 都有正确的复制行为，
        // 我们不需要额外的操作来实现深拷贝
    }

    // 赋值运算符
    Clause &Clause::operator=(const Clause &other)
    {
        if (this != &other)
        {
            literalMap = other.literalMap;
            literals = other.literals;
        }
        return *this;
    }

    // 移动构造函数
    Clause::Clause(Clause &&other) noexcept
        : literalMap(std::move(other.literalMap)),
          literals(std::move(other.literals))
    {
    }

    // 移动赋值运算符
    Clause &Clause::operator=(Clause &&other) noexcept
    {
        if (this != &other)
        {
            literalMap = std::move(other.literalMap);
            literals = std::move(other.literals);
        }
        return *this;
    }

    size_t Clause::hash() const
    {
        if (!hashComputed)
        {
            std::vector<size_t> literalHashes;
            for (const auto &lit : literals)
            {
                literalHashes.push_back(lit.hash());
            }
            std::sort(literalHashes.begin(), literalHashes.end());

            hashValue = 0;
            for (const auto &h : literalHashes)
            {
                hashValue ^= h + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
            }
            hashComputed = true;
        }
        return hashValue;
    }

    bool Clause::operator==(const Clause &other) const
    {
        if (literals.size() != other.literals.size())
            return false;

        std::unordered_multiset<Literal> thisLiterals(literals.begin(), literals.end());
        std::unordered_multiset<Literal> otherLiterals(other.literals.begin(), other.literals.end());

        return thisLiterals == otherLiterals;
    }
}