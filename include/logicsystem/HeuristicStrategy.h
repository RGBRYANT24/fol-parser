// HeuristicStrategy.h
#ifndef LOGIC_SYSTEM_HEURISTICSTRATEGY_STRATEGY_H
#define LOGIC_SYSTEM_HEURISTICSTRATEGY_STRATEGY_H

#include "ResolutionPair.h"
#include "SearchStrategy.h"
#include <queue>

namespace LogicSystem
{
    class HeuristicStrategy : public SearchStrategy
    {
    private:
        std::priority_queue<ResolutionPair> pq;

    public:
        // 添加显式的析构函数声明
        ~HeuristicStrategy() noexcept override = default;
        void addPair(const ResolutionPair &pair) override { pq.push(pair); }
        bool isEmpty() const override { return pq.empty(); }
        ResolutionPair getNext() override
        {
            auto pair = pq.top();
            pq.pop();
            return pair;
        }
    };
}

#endif // LOGIC_SYSTEM_HEURISTICSTRATEGY_STRATEGY_H