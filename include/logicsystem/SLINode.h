// SLINode.h
#ifndef LOGIC_SYSTEM_SLI_NODE_H
#define LOGIC_SYSTEM_SLI_NODE_H

#include "Literal.h"
#include "Unifier.h"
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace LogicSystem {
    using Substitution = std::unordered_map<SymbolId, SymbolId>;
    class SLINode {
    public:
        SLINode(const Literal& lit, bool isALiteral, int nodeId)
            : literal(lit)
            , is_A_literal(isALiteral)
            , node_id(nodeId)
            , is_active(true)
            , depth(0)
            , rule_applied("") {}  // 初始化rule_applied为空字符串

        Literal literal;
        bool is_A_literal;
        int node_id;
        bool is_active;
        int depth;
        std::string rule_applied;  // 添加规则信息成员
        
        std::weak_ptr<SLINode> parent;  // 使用weak_ptr避免循环引用
        std::vector<std::shared_ptr<SLINode>> children;
        
        // 用于substitution
        Substitution substitution;
    };
}

#endif // LOGIC_SYSTEM_SLI_NODE_H