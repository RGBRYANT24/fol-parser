#include "TruncateOperation.h"

namespace LogicSystem
{
    void TruncateOperation::undo()
    {
        for (size_t i = 0; i < affected_nodes.size(); ++i)
        {
            affected_nodes[i]->is_active = previous_states[i];
        }
    }
}