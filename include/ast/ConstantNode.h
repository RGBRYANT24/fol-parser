#ifndef CONSTANT_NODE_H
#define CONSTANT_NODE_H

#include "Node.h"

namespace AST
{
    class ConstantNode : public Node
    {
    public:
        //std::string name;

        ConstantNode(const std::string& n) {this -> Node::name = n;}
        inline void print() { std::cout << "Constant Node " << this -> Node::name << std::endl; }
        NodeType getType() const override { return CONSTANT; }
    };
}

#endif // CONSTANT_NODE_H