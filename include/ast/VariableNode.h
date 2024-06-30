#ifndef VARIABLE_NODE_H
#define VARIABLE_NODE_H

#include "Node.h"

namespace AST
{
    class VariableNode : public Node
    {
    public:
        // std::string name;
        // VariableNode(const std::string &n) : name(n) {}
        VariableNode(const std::string &n) { this->Node::name = n; }
        inline void print() { std::cout << "Variable Node " << this->name << std::endl; }
        // 在 VariableNode 类中添加：
        Node *clone() const override
        {
            return new VariableNode(this->name);
        }
        NodeType getType() const override { return VARIABLE; }
    };
}

#endif // VARIABLE_NODE_H