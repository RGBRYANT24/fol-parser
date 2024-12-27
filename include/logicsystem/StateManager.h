#ifndef LOGIC_SYSTEM_STATEMANAGER_H
#define LOGIC_SYSTEM_STATEMANAGER_H
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "SLITree.h"
#include "ProofState.h"

namespace LogicSystem
{
    class StateManager
    {
    public:
        using StatePtr = std::shared_ptr<ProofState>;

        StateManager() : next_state_id(0) {}

        // 生成并管理新状态
        StatePtr createState(std::unique_ptr<SLITree> tree,
                             const StatePtr &parent = nullptr,
                             const std::string &operation = "")
        {
            auto state = std::make_shared<ProofState>(
                std::move(tree),
                parent,
                operation);
            state->state_id = next_state_id++;
            all_states.push_back(state);
            return state;
        }

        // 获取所有生成的状态
        const std::vector<StatePtr> &getAllStates() const
        {
            return all_states;
        }

    private:
        std::vector<StatePtr> all_states; // 好像有点冗余 暂时先保留
        size_t next_state_id;
    };
}
#endif