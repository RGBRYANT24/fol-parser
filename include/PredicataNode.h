#ifndef PREDICATE_NODE_H
#define PREDICATE_NODE_H

#include "Node.h"

namespace AST{
    class PredicateNode : public Node {
public:
    std::string name;
    std::vector<std::shared_ptr<Node>> arguments; // Arguments could be vars, constants, functions

    PredicateNode(const std::string& n) : name(n) {}
    void print();
    NodeType getType() const override { return PREDICATE; }
};
}
#endif // PREDICATE_NODE_H