// Resolution.cpp
#include "Resolution.h"
#include "Unifier.h"
#include <algorithm>
#include <iostream>
namespace LogicSystem
{
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
        //pq = std::move(temp_pq);

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
        //pq = std::move(temp_pq);

        std::cout << "End of Priority Queue Contents" << std::endl;
    }

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
                            std::cout << "c1 " << c1.toString(kb) << " c2 " << c2.toString(kb) << std::endl;
                            double score = calculateHeuristic(c1, c2, l1, l2);
                            pq.emplace(&c1, &c2, l1, l2, score);
                        }
                    }
                }
            }
        }

        int count = 0;
        while (!pq.empty())
        {
            std::cout << "Round " << count + 1 << std::endl;

            ResolutionPair pair = pq.top();
            pq.pop();

            /*if (pair.clause1->isEmpty() && pair.clause2->isEmpty())
                return true;*/
            //std::cout << "To resovle: " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
            auto resolvant = resolve(*pair.clause1, *pair.clause2, pair.literal1Index, pair.literal2Index, kb);

            if (!resolvant)
            {
                //std::cout << "unresolvant with " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
                continue;
            }
            else
            {
                std::cout << "resolvant " << resolvant->toString(kb) << std::endl;
            }
            // break;
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
                        // if (resolvant->isEmpty())
                        //     std::cout << "add resolvant but empty" << std::endl;
                        // if (clause.isEmpty())
                        //     std::cout << "add clause but empty" << std::endl;
                        if (isComplementary(resolvant->getLiterals()[i], clause.getLiterals()[j]))
                        {
                            double score = calculateHeuristic(*resolvant, clause, i, j);
                            // std::cout << "score " << score << std::endl;
                            // std::cout << "new Complementary Pair " << resolvant->toString(kb) << " clause " << clause.toString(kb) << std::endl;
                            pq.emplace(&clauses.back(), &clause, static_cast<int>(i), static_cast<int>(j), score);
                        }
                    }
                }
            }
            //std::cout << pq.size() << " " << std::endl;
            //printPriorityQueue(pq, kb);
            if (count >= 4)
                break;
            count++;
        }

        return false; // 无法证明
    }

    
    bool Resolution::proveDFS(const KnowledgeBase &kb, const Clause &goal)
    {

        std::vector<Clause> clauses = kb.getClauses();
        clauses.push_back(goal);

        std::stack<ResolutionPair> stk;
        std::cout << "DFS" << std::endl;
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
                            std::cout << "c1 " << c1.toString(kb) << " c2 " << c2.toString(kb) << std::endl;
                            double score = calculateHeuristic(c1, c2, l1, l2);
                            stk.emplace(&c1, &c2, l1, l2, score);
                        }
                    }
                }
            }
        }

        int count = 0;
        while (!stk.empty())
        {
            std::cout << "Round " << count + 1 << std::endl;

            ResolutionPair pair = stk.top();
            stk.pop();

            /*if (pair.clause1->isEmpty() && pair.clause2->isEmpty())
                return true;*/
            //std::cout << "To resovle: " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
            auto resolvant = resolve(*pair.clause1, *pair.clause2, pair.literal1Index, pair.literal2Index, kb);

            if (!resolvant)
            {
                //std::cout << "unresolvant with " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
                continue;
            }
            else
            {
                std::cout << "resolvant " << resolvant->toString(kb) << std::endl;
            }
            // break;
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
                        // if (resolvant->isEmpty())
                        //     std::cout << "add resolvant but empty" << std::endl;
                        // if (clause.isEmpty())
                        //     std::cout << "add clause but empty" << std::endl;
                        if (isComplementary(resolvant->getLiterals()[i], clause.getLiterals()[j]))
                        {
                            double score = calculateHeuristic(*resolvant, clause, i, j);
                            // std::cout << "score " << score << std::endl;
                            // std::cout << "new Complementary Pair " << resolvant->toString(kb) << " clause " << clause.toString(kb) << std::endl;
                            stk.emplace(&clauses.back(), &clause, static_cast<int>(i), static_cast<int>(j), score);
                        }
                    }
                }
            }
            //std::cout << pq.size() << " " << std::endl;
            //printPriorityQueue(pq, kb);
            if (count >= 4)
                break;
            count++;
        }

        return false; // 无法证明
    }

    bool Resolution::proveBFS(const KnowledgeBase &kb, const Clause &goal)
    {

        std::vector<Clause> clauses = kb.getClauses();
        clauses.push_back(goal);
        std::queue<ResolutionPair> q;
        std::cout << "BFS" << std::endl;
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
                            std::cout << "c1 " << c1.toString(kb) << " c2 " << c2.toString(kb) << std::endl;
                            //double score = calculateHeuristic(c1, c2, l1, l2);
                            double score = 1;// BFS放的是在搜索树的层数， 1为起点
                            q.emplace(&c1, &c2, l1, l2, score);
                        }
                    }
                }
            }
        }

        int count = 0;
        while (!q.empty())
        {
            std::cout << "Round " << count + 1 << std::endl;
            std::cout << "queue before resolve " << std::endl;
            printQueue(q, kb);

            ResolutionPair pair = q.front();
            q.pop();

            /*if (pair.clause1->isEmpty() && pair.clause2->isEmpty())
                return true;*/
            //std::cout << "To resovle: " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
            auto resolvant = resolve(*pair.clause1, *pair.clause2, pair.literal1Index, pair.literal2Index, kb);

            if (!resolvant)
            {
                std::cout << "unresolvant with " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
                continue;
            }
            else
            {
                std::cout << "resolvant " << resolvant->toString(kb) << std::endl;
                std::cout << "Original Clauses: " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
            }
            // break;
            if (resolvant->isEmpty())
            {
                return true; // 找到空子句，证明成功
            }
            std::cout << "queue after resolve before push back " << std::endl;
            printQueue(q, kb);
            // 添加新的子句到 clauses
            clauses.push_back(*resolvant);
            std::cout << "Original Clauses after push_back: " << (*pair.clause1).toString(kb) << " index " << pair.literal1Index << " " << (*pair.clause2).toString(kb) << " index " << pair.literal2Index << std::endl;
            std::cout << "queue after push back resolve before emplace new" << std::endl;
            printQueue(q, kb);
            // 将新子句与现有子句进行比较
            for (const auto &clause : clauses)
            {
                for (size_t i = 0; i < resolvant->getLiterals().size(); ++i)
                {
                    for (size_t j = 0; j < clause.getLiterals().size(); ++j)
                    {
                        if (resolvant->isEmpty())
                            std::cout << "add resolvant but empty" << std::endl;
                        if (clause.isEmpty())
                            std::cout << "add clause but empty" << std::endl;
                        if (isComplementary(resolvant->getLiterals()[i], clause.getLiterals()[j]))
                        {
                            double score = pair.heuristicScore + 1;
                            // std::cout << "score " << score << std::endl;
                            // std::cout << "new Complementary Pair " << resolvant->toString(kb) << " clause " << clause.toString(kb) << std::endl;
                            q.emplace(&clauses.back(), &clause, static_cast<int>(i), static_cast<int>(j), score);
                        }
                    }
                }
            }
            std::cout << "queue after emplace " << std::endl;
            printQueue(q, kb);
            if (count >= 4)
                break;
            count++;
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