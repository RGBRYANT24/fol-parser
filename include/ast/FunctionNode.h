#ifndef FUNCTION_NODE_H
#define FUNCTION_NODE_H

#include "Node.h"
#include "TermListNode.h"

namespace AST{
    class FunctionNode : public Node {
public:
    //std::string name;
    //std::vector<Node*> arguments; // Arguments could be vars, constants
    AST::Node* termlists;
    FunctionNode();
    FunctionNode(const std::string& functionName, AST::Node* term_lists) : termlists(term_lists) {this -> Node::name = functionName;}
    bool insert(AST::Node* term);//add function arity in this->arguments
    void print();
    NodeType getType() const override { return FUNCTION; }
    ~FunctionNode();
};
}
#endif // FUNCTION_NODE_H