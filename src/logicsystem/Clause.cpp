#include "Clause.h"
#include "KnowledgeBase.h"
#include <iostream>

namespace LogicSystem
{
    void Clause::addLiteral(const Literal &lit)
    {
        int predicateId = lit.getPredicateId();
        //std::cout << "PredicatedId " << predicateId << std::endl;

        if (this->hasOppositeLiteral(lit)) // 如果存在互补的文字,不添加,直接删除
        {
            auto it = this->literalMap.find(predicateId);
            if (it != this->literalMap.end())
            {
                int indexToRemove = it->second;
                // 将要删除的元素与最后一个元素交换，然后删除最后一个元素
                if (indexToRemove < literals.size() - 1)
                {
                    std::swap(literals[indexToRemove], literals.back());
                    // 更新被交换到indexToRemove位置的文字在literalMap中的索引
                    literalMap[literals[indexToRemove].getPredicateId()] = indexToRemove;
                }
                literals.pop_back();
                this->literalMap.erase(it);
            }
        }
        else if (this->literalMap.find(predicateId) != this->literalMap.end()) // 如果已经有相同项，不操作
        {
            return;
        }
        else
        {
            literals.push_back(lit);
            this->literalMap[predicateId] = this->literals.size() - 1;
        }
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
        if (this->literalMap.find(nameId) != this->literalMap.end() && this->literals[this->literalMap.at(nameId)].isNegated() != lit.isNegated())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // 复制构造函数
    Clause::Clause(const Clause& other)
        : literalMap(other.literalMap), literals(other.literals)
    {
        // 由于 std::unordered_map 和 std::vector 都有正确的复制行为，
        // 我们不需要额外的操作来实现深拷贝
    }

    // 赋值运算符
    Clause& Clause::operator=(const Clause& other)
    {
        if (this != &other)
        {
            literalMap = other.literalMap;
            literals = other.literals;
        }
        return *this;
    }

    // 移动构造函数
    Clause::Clause(Clause&& other) noexcept
        : literalMap(std::move(other.literalMap)), 
          literals(std::move(other.literals))
    {
    }

    // 移动赋值运算符
    Clause& Clause::operator=(Clause&& other) noexcept
    {
        if (this != &other)
        {
            literalMap = std::move(other.literalMap);
            literals = std::move(other.literals);
        }
        return *this;
    }
}