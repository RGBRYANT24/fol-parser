// DFSStrategy.h
#ifndef LOGIC_SYSTEM_DFS_STRATEGY_H
#define LOGIC_SYSTEM_DFS_STRATEGY_H

#include "ResolutionPair.h"
#include "SearchStrategy.h"
#include <stack>

namespace LogicSystem
{
    class DFSStrategy : public SearchStrategy
    {
    private:
        std::stack<ResolutionPair> s;

    public:
        void addPair(const ResolutionPair &pair) override { s.push(pair); }
        bool isEmpty() const override { return s.empty(); }
        ResolutionPair getNext() override
        {
            auto pair = s.top();
            s.pop();
            return pair;
        }
    };
}

#endif //LOGIC_SYSTEM_DFS_STRATEGY_H