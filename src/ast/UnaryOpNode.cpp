#include "UnaryOpNode.h"

AST::UnaryOpNode::UnaryOpNode()
{
    this -> op = AST::Node::NOT;
    this -> child = nullptr;
}

void AST::UnaryOpNode::print()
{
    std::cout << "Unary Node Print" << std::endl;
    this -> child -> print();
}

bool AST::UnaryOpNode::insert(AST::Node* term)
{
    return this -> child -> insert(term);
}

AST::UnaryOpNode::~UnaryOpNode()
{
    delete this -> child;
    std::cout << "Unary Node Destroy " << std::endl;
}