#ifndef LOGIC_SYSTEM_PROOFSTATE_H
#define LOGIC_SYSTEM_PROOFSTATE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "SLITree.h"

namespace LogicSystem
{
    struct ProofState
    {
        static int next_id;
        int state_id;
        std::shared_ptr<ProofState> parent;
        std::unique_ptr<SLITree> tree;

        // 添加一个标记变量，记录当前状态应用了哪个规则
        std::string applied_rule;

        ProofState(std::unique_ptr<SLITree> t,
                   std::shared_ptr<ProofState> p = nullptr,
                   std::string rule = "")
            : tree(std::move(t)),
              parent(p),
              applied_rule(rule)
        {
            state_id = next_id++;
        }
    };
}
#endif