// SLINode.h
#ifndef LOGIC_SYSTEM_SLI_NODE_H
#define LOGIC_SYSTEM_SLI_NODE_H

#include "Literal.h"
#include "Unifier.h"
#include "KnowledgeBase.h"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <atomic>

namespace LogicSystem
{
    using Substitution = std::unordered_map<SymbolId, SymbolId>;
    class SLINode
    {
    public:

        static std::atomic<int> next_node_id;  // 使用原子计数器确保唯一ID
        SLINode(const Literal &lit, bool isALiteral, int nodeId);

        Literal literal;
        bool is_A_literal;
        int node_id;
        bool is_active;
        int depth;
        std::string rule_applied;

        std::weak_ptr<SLINode> parent;
        std::vector<std::shared_ptr<SLINode>> children;

        Substitution substitution;

        // 声明print方法
        void print(const KnowledgeBase &kb) const;

    private:
        static std::string join(const std::vector<std::string> &elements, const std::string &delimiter);
    };
    
}

#endif // LOGIC_SYSTEM_SLI_NODE_H