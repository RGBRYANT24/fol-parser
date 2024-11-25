#ifndef LOGIC_SYSTEM_ADD_OPERATION_H
#define LOGIC_SYSTEM_ADD_OPERATION_H

#include "Operation.h"
#include "SLINode.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace LogicSystem
{
    class AddOperation : public Operation
    {
    public:
        AddOperation(const std::vector<std::shared_ptr<SLINode>> &nodes,
                    std::unordered_map<size_t, std::shared_ptr<SLINode>> &litMap,
                    std::vector<std::vector<std::shared_ptr<SLINode>>> &depthMap)
            : added_nodes(nodes), literal_map(litMap), depth_map(depthMap) {}

        void undo() override;

    private:
        std::vector<std::shared_ptr<SLINode>> added_nodes;
        std::unordered_map<size_t, std::shared_ptr<SLINode>> &literal_map;
        std::vector<std::vector<std::shared_ptr<SLINode>>> &depth_map;
    };
}

#endif // LOGIC_SYSTEM_ADD_OPERATION_H