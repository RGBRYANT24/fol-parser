// SLINode.cpp
#include "SLINode.h"
#include <iostream>

namespace LogicSystem {

std::atomic<int> SLINode::next_node_id{0};  // 从0开始初始化

SLINode::SLINode(const Literal& lit, bool isALiteral, int nodeId)
    : literal(lit)
    , is_A_literal(isALiteral)
    , node_id(nodeId)
    , is_active(true)
    , depth(0)
    , rule_applied("") {}

std::string SLINode::join(const std::vector<std::string>& elements, const std::string& delimiter) {
    std::string result;
    bool first = true;
    for (const auto& element : elements) {
        if (!first) {
            result += delimiter;
        }
        result += element;
        first = false;
    }
    return result;
}

void SLINode::print(const KnowledgeBase& kb) const {
    if (literal.isEmpty()) {
        std::cout << "ROOT" << std::endl;
        return;
    }

    // 文字信息
    std::cout << literal.toString(kb)
              << (is_A_literal ? "*" : "");

    // 节点基本信息
    std::cout << " [" << node_id << "|d:" << depth << "]";

    // 替换信息
    if (!substitution.empty()) {
        std::cout << " subst:{";
        bool first = true;
        for (const auto& [var, term] : substitution) {
            if (!first)
                std::cout << ",";
            std::cout << kb.getSymbolName(var) << "/" << kb.getSymbolName(term);
            first = false;
        }
        std::cout << "}";
    }

    // 状态信息
    std::cout << " (";
    std::vector<std::string> status;
    if (!is_active)
        status.push_back("inactive");
    if (is_A_literal)
        status.push_back("A-lit");
    if (status.empty())
        status.push_back("active");
    std::cout << join(status, ",") << ")";

    // 其他调试信息
    if (auto p = parent.lock()) {
        std::cout << " parent:" << p->node_id;
    }
    std::cout << " children:" << children.size();

    // 规则信息
    if (!rule_applied.empty()) {
        std::cout << " rule:" << rule_applied;
    }

    std::cout << "\n";
}

} // namespace LogicSystem