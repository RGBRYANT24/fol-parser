#ifndef PREDICATE_NODE_H
#define PREDICATE_NODE_H

#include "Node.h"

namespace AST{
    class PredicateNode : public Node {
public:
    std::string name;
    //std::vector<AST::Node*> arguments; // Arguments could be vars, constants, functions
    AST::Node* termlists;
    PredicateNode();
    PredicateNode(const std::string& n) : name(n), termlists(nullptr) {}
    PredicateNode(const std::string& n, AST::Node* term_lists) : name(n), termlists(term_lists) {}
    void print();
    bool insert(AST::Node* term);
    NodeType getType() const override { return PREDICATE; }
    ~PredicateNode();
};
}
#endif // PREDICATE_NODE_H