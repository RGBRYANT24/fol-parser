#ifndef UNARY_OP_NODE_H
#define UNARY_OP_NODE_H

#include "Node.h"

namespace AST {
    class UnaryOpNode : public Node {
    public:
        AST::Node* child;
        AST::Node::NodeType op;

        UnaryOpNode();
        UnaryOpNode(AST::Node::NodeType op, AST::Node* child) : op(op), child(child) {}
        void print() override;
        bool insert(AST::Node* term) override;
        NodeType getType() const override {return this -> op;}
        ~UnaryOpNode();
    };
}

#endif // UNARY_OP_NODE_H