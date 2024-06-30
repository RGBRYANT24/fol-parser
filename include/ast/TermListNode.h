#ifndef TERMLIST_NODE_H
#define TERMLIST_NODE_H

#include "Node.h"

namespace AST{
    class TermListNode : public Node {
public:
    //std::string name;
    std::vector<Node*> arguments; // Arguments could be vars, constants, functions

    TermListNode() {}
    AST::Node* clone() const override;
    bool insert(AST::Node* term);
    void print();
    NodeType getType() const override { return TERMLIST; }
    ~TermListNode();
};
}
#endif // PREDICATE_NODE_H