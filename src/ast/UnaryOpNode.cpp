#include "UnaryOpNode.h"

AST::UnaryOpNode::UnaryOpNode()
{
    this->op = AST::Node::NOT;
    this->child = nullptr;
}

void AST::UnaryOpNode::print()
{
    std::cout << "Unary Node Print" << std::endl;
    this->child->print();
}

bool AST::UnaryOpNode::insert(AST::Node *term)
{
    return this->child->insert(term);
}

AST::Node *AST::UnaryOpNode::clone() const
{
    UnaryOpNode *newNode = new UnaryOpNode();
    newNode->op = this->op;
    newNode->name = this->name;

    if (this->child)
    {
        newNode->child = this->child->clone();
    }
    else
    {
        newNode->child = nullptr;
    }

    return newNode;
}

AST::UnaryOpNode::~UnaryOpNode()
{
    delete this->child;
    std::cout << "Unary Node Destroy " << std::endl;
}