#ifndef CONSTANT_NODE_H
#define CONSTANT_NODE_H

#include "Node.h"

namespace AST
{
    class ConstantNode : public Node
    {
    public:
        std::string name;

        ConstantNode(const std::string& n) : name(n) {}
        inline void print() { std::cout << "Constant Node " << this->name << std::endl; }
        NodeType getType() const override { return CONSTANT; }
    };
}

#endif // CONSTANT_NODE_H