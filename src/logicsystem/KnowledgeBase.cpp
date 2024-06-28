#include "KnowledgeBase.h"
#include <iostream>
#include <algorithm>

namespace LogicSystem
{

    void KnowledgeBase::addClause(Clause *clause)
    {
        this->clauseOrder.push_back(clause);
        this->indexClause(clause);
    }

    void KnowledgeBase::removeClause(Clause *clause)
    {
        this->deindexClause(clause);
        //"erase-remove idiom"
        // remove 将所有不等于 clause 的元素移到 vector 的前面,返回一个迭代器，指向最后一个不应被删除的元素之后的位置
        // erase 方法真正从 vector 中删除元素,它删除从 std::remove 返回的迭代器位置到 vector 末尾的所有元素
        // 能在一次操作中高效地删除 vector 中的所有指定元素。时间复杂度为 O(n)，其中 n 是 vector 的大小
        this->clauseOrder.erase(std::remove(this->clauseOrder.begin(), this->clauseOrder.end(), clause), this->clauseOrder.end());
        delete clause;
    }

    std::vector<Clause *> KnowledgeBase::getClauses() const
    {
        return this->clauseOrder;
    }

    std::vector<Clause *> KnowledgeBase::getClausesWithPredicate(const std::string &predicateName) const
    {
        auto it = this->predicateIndex.find(predicateName);
        if (it != this->predicateIndex.end())
        {
            return std::vector<Clause *>(it->second.begin(), it->second.end());
        }
        return std::vector<Clause *>();
    }

    void KnowledgeBase::print() const
    {
        std::cout << "Knowledge Base Print Begin-------------------" << std::endl;
        for (size_t i = 0; i < this->clauseOrder.size(); ++i)
        {
            std::cout << "Clause " << i + 1 << ": ";
            this->clauseOrder[i]->print();
            std::cout << std::endl;
        }
        std::cout << "Knowledge Base Print End-------------------" << std::endl;
    }

    size_t KnowledgeBase::size() const
    {
        return this->clauseOrder.size();
    }

    KnowledgeBase::~KnowledgeBase()
    {
        for (auto clause : this->clauseOrder)
        {
            delete clause;
        }
    }

    void KnowledgeBase::indexClause(Clause *clause)
    {
        for (const auto &literal : clause->getAllLiterals())
        {
            this->predicateIndex[literal->getPredicateName()].insert(clause);
        }
    }

    void KnowledgeBase::deindexClause(Clause *clause)
    {
        for (const auto &literal : clause->getAllLiterals())
        {
            auto &clauses = this->predicateIndex[literal->getPredicateName()];
            clauses.erase(clause);
            if (clauses.empty())
            {
                this->predicateIndex.erase(literal->getPredicateName());
            }
        }
    }
}