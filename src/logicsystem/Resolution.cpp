// Resolution.cpp
#include "Resolution.h"
#include "Unifier.h"
#include <algorithm>
#include <iostream>
#include <memory>
namespace LogicSystem
{
    struct ClauseHash
    {
        size_t operator()(const std::shared_ptr<Clause> &clause) const
        {
            return clause->hash();
        }
    };

    struct ClauseEqual
    {
        bool operator()(const std::shared_ptr<Clause> &lhs, const std::shared_ptr<Clause> &rhs) const
        {
            return *lhs == *rhs;
        }
    };
    std::vector<std::shared_ptr<Clause>> convertToSharedPtr(const std::vector<Clause> &originalClauses)
    {
        std::vector<std::shared_ptr<Clause>> newClauses;
        newClauses.reserve(originalClauses.size());
        for (const auto &clause : originalClauses)
        {
            newClauses.push_back(std::make_shared<Clause>(clause));
        }
        return newClauses;
    }
    std::vector<std::unique_ptr<Clause>> convertToUniquePtr(const std::vector<Clause> &originalClauses)
    {
        std::vector<std::unique_ptr<Clause>> newClauses;
        newClauses.reserve(originalClauses.size());
        for (const auto &clause : originalClauses)
        {
            newClauses.push_back(std::make_unique<Clause>(clause));
        }
        return newClauses;
    }
    void printPriorityQueue(const std::priority_queue<ResolutionPair> &pq, const KnowledgeBase &kb)
    {
        std::cout << "Priority Queue Contents:" << std::endl;
        std::cout << "Total elements: " << pq.size() << std::endl;

        // 创建一个临时队列
        std::priority_queue<ResolutionPair> temp_pq = pq;

        // 遍历并打印优先队列中的所有元素
        int count = 0;
        while (!temp_pq.empty())
        {
            const ResolutionPair &pair = temp_pq.top();
            temp_pq.pop();
            std::cout << "Element " << count++ << ":" << std::endl;
            std::cout << "  Clause1: " << pair.clause1->toString(kb) << std::endl;
            std::cout << "  Clause2: " << pair.clause2->toString(kb) << std::endl;
            std::cout << "  Opposite Literal Indices: " << pair.literal1Index << ", " << pair.literal2Index << std::endl;
            std::cout << "  Score: " << pair.heuristicScore << std::endl;

            // // 将元素添加到临时队列
            // temp_pq.push(pair);
            // pq.pop();
        }

        // 将所有元素放回原始优先队列
        // pq = std::move(temp_pq);

        std::cout << "End of Priority Queue Contents" << std::endl;
    }
    void printQueue(const std::queue<ResolutionPair> &q, const KnowledgeBase &kb)
    {
        std::cout << "Priority Queue Contents:" << std::endl;
        std::cout << "Total elements: " << q.size() << std::endl;

        // 创建一个临时队列
        std::queue<ResolutionPair> temp_q = q;

        // 遍历并打印优先队列中的所有元素
        int count = 0;
        while (!temp_q.empty())
        {
            const ResolutionPair &pair = temp_q.front();
            temp_q.pop();
            std::cout << "Element " << count++ << ":" << std::endl;
            std::cout << "  Clause1: " << pair.clause1->toString(kb) << std::endl;
            std::cout << "  Clause2: " << pair.clause2->toString(kb) << std::endl;
            std::cout << "  Opposite Literal Indices: " << pair.literal1Index << ", " << pair.literal2Index << std::endl;
            std::cout << "  Score: " << pair.heuristicScore << std::endl;

            // // 将元素添加到临时队列
            // temp_pq.push(pair);
            // pq.pop();
        }

        // 将所有元素放回原始优先队列
        // pq = std::move(temp_pq);

        std::cout << "End of Priority Queue Contents" << std::endl;
    }

    bool Resolution::prove(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy, int addClauseSize)
    {
        std::vector<std::shared_ptr<Clause>> clauses = convertToSharedPtr(kb.getClauses());
        clauses.push_back(std::make_shared<Clause>(goal));
        std::unordered_set<std::shared_ptr<Clause>, ClauseHash, ClauseEqual> visitedClauses;

        // 初始化搜索策略
        for (size_t i = 0; i < clauses.size(); ++i)
        {
            for (size_t j = i + 1; j < clauses.size(); ++j)
            {
                const auto &c1 = clauses[i];
                const auto &c2 = clauses[j];
                for (size_t l1 = 0; l1 < c1->getLiterals().size(); ++l1)
                {
                    for (size_t l2 = 0; l2 < c2->getLiterals().size(); ++l2)
                    {
                        if (isComplementary(c1->getLiterals()[l1], c2->getLiterals()[l2]))
                        {
                            double score = calculateHeuristic(*c1, *c2, l1, l2);
                            strategy.addPair(ResolutionPair(c1, c2, l1, l2, score));
                        }
                    }
                }
            }
        }

        int count = 0;
        while (!strategy.isEmpty())
        {
            count++;
            if (count % 10000 == 1)
            {
                std::cout << "Round " << count << std::endl;
            }

            ResolutionPair pair = strategy.getNext();
            auto resolvant = resolve(*pair.clause1, *pair.clause2, pair.literal1Index, pair.literal2Index, kb);

            if (!resolvant)
                continue;
            if (resolvant->isEmpty())
                return true;
            if (resolvant->isTautology() || resolvant->containsSelfLoop(kb))
                continue;

            auto newClause = std::make_shared<Clause>(*resolvant);
            if (visitedClauses.find(newClause) != visitedClauses.end())
                continue;

            visitedClauses.insert(newClause);

            if (resolvant->getLiterals().size() <= addClauseSize)
            {
                clauses.push_back(newClause);
            }

            for (const auto &clause : clauses)
            {
                for (size_t i = 0; i < newClause->getLiterals().size(); ++i)
                {
                    for (size_t j = 0; j < clause->getLiterals().size(); ++j)
                    {
                        if (isComplementary(newClause->getLiterals()[i], clause->getLiterals()[j]))
                        {
                            double score = calculateHeuristic(*newClause, *clause, i, j);
                            strategy.addPair(ResolutionPair(newClause, clause, i, j, score));
                        }
                    }
                }
            }

            if (count >= 1e11)
                break;
        }

        return false;
    }

    double Resolution::calculateHeuristic(const Clause &c1, const Clause &c2, int l1, int l2)
    {
        // 这里是一个简单的启发式函数，后续可以用神经网络替换
        return c1.getLiterals().size() + c2.getLiterals().size() - 1;
    }

    std::optional<Clause> Resolution::resolve(const Clause &c1, const Clause &c2, int l1, int l2, KnowledgeBase &kb)
    {
        if (l1 < 0 || l1 >= c1.getLiterals().size() || l2 < 0 || l2 >= c2.getLiterals().size())
        {
            return std::nullopt;
        }
        const Literal &lit1 = c1.getLiterals()[l1];
        const Literal &lit2 = c2.getLiterals()[l2];

        auto mgu = Unifier::findMGU(lit1, lit2, kb);
        if (!mgu)
        {
            return std::nullopt;
        }

        // std::cout << "Calculate MGU " << std::endl;
        // std::cout << "c1 " << c1.toString(kb) << " c2 " << c2.toString(kb) << std::endl;
        // std::cout << " MGU " << std::endl;
        // for (const auto &[key, value] : *mgu)
        // {
        //     std::cout << "  " << kb.getSymbolName(key) << " -> " << kb.getSymbolName(value) << std::endl;
        // }

        Clause resolvant;
        // std::cout << "resolvant pinjie add c1" << std::endl;
        for (const auto &lit : c1.getLiterals())
        {
            // auto clause1 = c1.getLiterals();
            if (&lit != &c1.getLiterals()[l1])
            {
                LogicSystem::Literal newLiteral = Unifier::applySubstitutionToLiteral(lit, *mgu, kb);
                resolvant.addLiteral(newLiteral);
                /*std::cout << "literal " << newLiteral.toString(kb) << " predicate id " << newLiteral.getPredicateId() << std::endl;
                std::cout << "resolvant after add this literal " << resolvant.toString(kb) << std::endl;*/
                // resolvant.addLiteral(Unifier::applySubstitutionToLiteral(lit, *mgu, kb));
            }
        }
        // std::cout <<"resolvant pinjie add c2 " << std::endl;
        for (const auto &lit : c2.getLiterals())
        {
            if (&lit != &c2.getLiterals()[l2])
            {
                LogicSystem::Literal newLiteral = Unifier::applySubstitutionToLiteral(lit, *mgu, kb);
                resolvant.addLiteral(newLiteral);
                /*std::cout << "literal " << newLiteral.toString(kb) << " predicate id " << newLiteral.getPredicateId() << std::endl;
                std::cout << "resolvant after add this literal " << resolvant.toString(kb) << std::endl;*/
                // resolvant.addLiteral(Unifier::applySubstitutionToLiteral(lit, *mgu, kb));
            }
        }
        // resolvant.check()//这里面clause应该做检查 如果有互补的文字项直接删除，相同的文字项只保留一个
        return resolvant;
    }

    bool Resolution::isComplementary(const Literal &lit1, const Literal &lit2)
    {
        return lit1.getPredicateId() == lit2.getPredicateId() &&
               lit1.getArgumentIds().size() == lit2.getArgumentIds().size() && // 不检查参数完全一致 就会越界
               lit1.isNegated() != lit2.isNegated();
    }

} // namespace LogicSystem