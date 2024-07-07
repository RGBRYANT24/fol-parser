// Resolution.cpp
#include "Resolution.h"
#include "Unifier.h"
#include <algorithm>
#include <iostream>
namespace LogicSystem
{

    bool Resolution::prove(const KnowledgeBase &kb, const Clause &goal)
    {
        std::vector<Clause> clauses = kb.getClauses();
        clauses.push_back(goal);

        std::priority_queue<ResolutionPair> pq;

        // 初始化优先队列
        for (size_t i = 0; i < clauses.size(); ++i)
        {
            for (size_t j = i + 1; j < clauses.size(); ++j)
            {
                const Clause &c1 = clauses[i];
                const Clause &c2 = clauses[j];

                for (size_t l1 = 0; l1 < c1.getLiterals().size(); ++l1)
                {
                    for (size_t l2 = 0; l2 < c2.getLiterals().size(); ++l2)
                    {
                        if (isComplementary(c1.getLiterals()[l1], c2.getLiterals()[l2]))
                        {
                            std::cout << c1.toString(kb) << " " << c2.toString(kb) << std::endl;
                            double score = calculateHeuristic(c1, c2, l1, l2);
                            pq.emplace(&c1, &c2, l1, l2, score);
                        }
                    }
                }
            }
        }

        while(!pq.empty())
        {
            ResolutionPair pair = pq.top();
            pq.pop();

            auto resolvant = resolve(*pair.clause1, *pair.clause2, pair.literal1Index, pair.literal2Index, kb);

            if (!resolvant)
            {
                continue;
            }

            if (resolvant->isEmpty())
            {
                return true; // 找到空子句，证明成功
            }

            // 添加新的子句到 clauses
            clauses.push_back(*resolvant);

            // 将新子句与现有子句进行比较
            for (const auto &clause : clauses)
            {
                for (size_t i = 0; i < resolvant->getLiterals().size(); ++i)
                {
                    for (size_t j = 0; j < clause.getLiterals().size(); ++j)
                    {
                        if (isComplementary(resolvant->getLiterals()[i], clause.getLiterals()[j]))
                        {
                            double score = calculateHeuristic(*resolvant, clause, i, j);
                            pq.emplace(&clauses.back(), &clause, static_cast<int>(i), static_cast<int>(j), score);
                        }
                    }
                }
            }
        }

        return false; // 无法证明
    }

    double Resolution::calculateHeuristic(const Clause &c1, const Clause &c2, int l1, int l2)
    {
        // 这里是一个简单的启发式函数，后续可以用神经网络替换
        return c1.getLiterals().size() + c2.getLiterals().size() - 1;
    }

    std::optional<Clause> Resolution::resolve(const Clause &c1, const Clause &c2, int l1, int l2, const KnowledgeBase &kb)
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

        std::cout << "Calculate MGU " << std::endl;
        std::cout << "c1 " << c1.toString(kb) << " c2 " << c2.toString(kb) << std::endl;
        std::cout << " MGU " << std::endl;
        for (const auto &[key, value] : *mgu)
        {
            std::cout << "  " << kb.getVariableName(key) << " -> " << kb.getVariableName(value) << std::endl;
        }

        Clause resolvant;
        for (const auto &lit : c1.getLiterals())
        {
            if (&lit != &c1.getLiterals()[l1])
            {
                resolvant.addLiteral(Unifier::applySubstitutionToLiteral(lit, *mgu, kb));
            }
        }
        for (const auto &lit : c2.getLiterals())
        {
            if (&lit != &c2.getLiterals()[l2])
            {
                resolvant.addLiteral(Unifier::applySubstitutionToLiteral(lit, *mgu, kb));
            }
        }

        return resolvant;
    }

    bool Resolution::isComplementary(const Literal &lit1, const Literal &lit2)
    {
        return lit1.getPredicateId() == lit2.getPredicateId() &&
               lit1.getArgumentIds().size() == lit2.getArgumentIds().size() && // 不检查参数完全一致 就会越界
               lit1.isNegated() != lit2.isNegated();
    }

} // namespace LogicSystem