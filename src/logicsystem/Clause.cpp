#include "Clause.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    void Clause::addLiteral(const Literal &lit)
    {
        int predicateId = lit.getPredicateId();

        if(this->hasOppositeLiteral(lit))//如果存在互补的文字,不添加,直接删除
        {
            int oppositeKey = this->literalMap[predicateId];
            this->literalMap.erase(predicateId);
            this->literals.erase(this->literals.begin() + predicateId);
        }
        else if(this -> literalMap.find(predicateId) != this -> literalMap.end())//如果已经有相同项，不操作
        {
            return;
        }
        else
        {
            literals.push_back(lit);
            this->literalMap[predicateId] = this->literals.size()-1;
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
        //如果出现相同谓词并且互补
        if(this -> literalMap.find(nameId) != this -> literalMap.end() && this -> literals[nameId].isNegated() != lit.isNegated())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}