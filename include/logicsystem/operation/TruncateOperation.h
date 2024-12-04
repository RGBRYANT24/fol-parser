// TruncateOperation.h
#ifndef LOGIC_SYSTEM_TRUNCATE_OPERATION_H
#define LOGIC_SYSTEM_TRUNCATE_OPERATION_H

#include "Operation.h"
#include "SLINode.h"
#include <memory>
#include <vector>

namespace LogicSystem
{
    class TruncateOperation : public Operation
    {
    public:
        TruncateOperation() = default;

        void save_state(std::shared_ptr<SLINode> node) {
            affected_nodes.push_back(node);
            previous_states.push_back(node->is_active);
            previous_rules.push_back(node->rule_applied);
        }

        void undo() override {
            for (size_t i = 0; i < affected_nodes.size(); ++i) {
                affected_nodes[i]->is_active = previous_states[i];
                affected_nodes[i]->rule_applied = previous_rules[i];
            }
        }

    private:
        std::vector<std::shared_ptr<SLINode>> affected_nodes;
        std::vector<bool> previous_states;
        std::vector<std::string> previous_rules;
    };
}

#endif // LOGIC_SYSTEM_TRUNCATE_OPERATION_H