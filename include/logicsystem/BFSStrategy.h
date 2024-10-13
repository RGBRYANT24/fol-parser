// BFSStrategy.h
#ifndef LOGIC_SYSTEM_BFS_STRATEGY_H
#define LOGIC_SYSTEM_BFS_STRATEGY_H

#include "ResolutionPair.h"
#include "SearchStrategy.h"
#include <queue>
namespace LogicSystem
{
    class BFSStrategy : public SearchStrategy
    {
    private:
        std::queue<ResolutionPair> q;

    public:
        void addPair(const ResolutionPair &pair) override { q.push(pair); }
        bool isEmpty() const override { return q.empty(); }
        ResolutionPair getNext() override
        {
            auto pair = q.front();
            q.pop();
            return pair;
        }
    };
}

#endif // LOGIC_SYSTEM_BFS_STRATEGY_H