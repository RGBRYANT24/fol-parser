#include "AddOperation.h"
#include <algorithm>

namespace LogicSystem
{
    void AddOperation::undo()
    {
        for (const auto &node : added_nodes)
        {
            // 从父节点的children中移除
            if (auto parent = node->parent.lock())
            {
                auto &children = parent->children;
                children.erase(
                    std::remove(children.begin(), children.end(), node),
                    children.end());
            }

            // 从深度图中移除
            if (node->depth < depth_map.size())
            {
                auto &depth_level = depth_map[node->depth];
                depth_level.erase(
                    std::remove(depth_level.begin(), depth_level.end(), node),
                    depth_level.end());
            }

            // 从文字映射中移除
            literal_map.erase(node->literal.hash());
        }
    }
}