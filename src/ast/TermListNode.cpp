#include "TermListNode.h"

bool AST::TermListNode::insert(AST::Node *term)
{
    for (const auto node : this->arguments)
    {
        if (term == node)
        {
            std::cout << "Insert term failed, already exists in arguments" << std::endl;
            // term -> print();
            return false;
        }
    }
    this->arguments.push_back(term);
    return true;
}

void AST::TermListNode::print()
{
    std::cout << "Term List Node print" << std::endl;
    for (const auto node : this->arguments)
    {
        node->print();
    }
}

AST::TermListNode::~TermListNode()
{
    for (Node *node : this->arguments)
    {
        delete node;
    }
    arguments.clear();
    std::cout << "Term List Node Destroy" << std::endl;
}