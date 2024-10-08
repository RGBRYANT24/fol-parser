#include "Literal.h"
#include "KnowledgeBase.h"
#include <iostream>

namespace LogicSystem
{
    Literal::Literal(int predId, const std::vector<SymbolId> &argIds, bool negated)
        : predicateId(predId), argumentIds(argIds), negated(negated) {}

    std::string Literal::toString(const KnowledgeBase &kb) const
    {
        std::string result = negated ? "¬" : "";
        result += kb.getPredicateName(predicateId) + "(";
        for (size_t i = 0; i < argumentIds.size(); ++i)
        {
            const SymbolId &symbolId = argumentIds[i];
            result += kb.getSymbolName(symbolId);
            if (i < argumentIds.size() - 1)
            {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }

    // 添加 == 运算符
    bool Literal::operator==(const Literal &other) const
    {
        // 1. 检查极性是否相同
        if (this->negated != other.negated)
        {
            return false;
        }

        // 2. 检查谓词 ID 是否相同
        if (this->predicateId != other.predicateId)
        {
            return false;
        }

        // 3. 检查参数数量是否相同
        if (this->argumentIds.size() != other.argumentIds.size())
        {
            return false;
        }

        // 4. 逐一比较每个参数，包括变量的 ID
        for (size_t i = 0; i < this->argumentIds.size(); ++i)
        {
            if (this->argumentIds[i] != other.argumentIds[i])
            {
                return false;
            }
        }

        // 所有检查都通过，两个 Literal 相等
        return true;
    }

    // 添加 != 运算符
    bool Literal::operator!=(const Literal &other) const
    {
        return !(*this == other);
    }

    int Literal::getPredicateId() const
    {
        return predicateId;
    }

    const std::vector<SymbolId> &Literal::getArgumentIds() const
    {
        return argumentIds;
    }

    bool Literal::isNegated() const
    {
        return negated;
    }
}