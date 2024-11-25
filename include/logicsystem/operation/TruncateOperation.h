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
        explicit TruncateOperation(std::shared_ptr<SLINode> n) : node(n) {}
        void save_state(std::shared_ptr<SLINode> n)
        {
            affected_nodes.push_back(n);
            previous_states.push_back(n->is_active);
        }
        void undo() override;

    private:
        std::shared_ptr<SLINode> node;
        std::vector<bool> previous_states;
        std::vector<std::shared_ptr<SLINode>> affected_nodes;
    };
}

#endif // LOGIC_SYSTEM_TRUNCATE_OPERATION_H