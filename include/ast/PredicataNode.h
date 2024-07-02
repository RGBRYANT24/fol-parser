#ifndef PREDICATE_NODE_H
#define PREDICATE_NODE_H

#include "Node.h"

namespace AST{
    class PredicateNode : public Node {
public:
    //std::string name;
    
    //std::vector<AST::Node*> arguments; // Arguments could be vars, constants, functions
    AST::Node* termlists;
    PredicateNode();
    PredicateNode(const std::string& n) : termlists(nullptr) {this -> Node::name = n;}
    PredicateNode(const std::string& n, AST::Node* term_lists) : termlists(term_lists) {this -> Node::name = n;}
    AST::Node* clone() const override;
    void print();
    bool insert(AST::Node* term);
    NodeType getType() const override { return PREDICATE; }
    ~PredicateNode();
};
}
#endif // PREDICATE_NODE_H