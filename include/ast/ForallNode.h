#ifndef FORALL_NODE_H
#define FORALL_NODE_H

#include "Node.h"

namespace AST
{
    class ForallNode : public Node
    {
    public:
        std::shared_ptr<Node> variable;
        std::shared_ptr<Node> formula;
        //std::string name;

        ForallNode(std::shared_ptr<Node> var, std::shared_ptr<Node> form) : variable(var), formula(form) {}//need fix
        inline void print() { std::cout << "Forall Node " << this -> Node::name << std::endl; }
        NodeType getType() const override { return FORALL; }
    };
}

#endif // FORALL_NODE_H