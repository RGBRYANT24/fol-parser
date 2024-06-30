#include "PredicataNode.h"

AST::PredicateNode::PredicateNode()
{
    this->Node::name = "";
    this->termlists = nullptr;
}

void AST::PredicateNode::print()
{
    std::cout << "PredicateNode name " << this->Node::name << " arguments: ";
    if (this->termlists == nullptr)
    {
        std::cout << "No Predicate term lists " << std::endl;
        return;
    }
    else
    {
        this->termlists->print();
    }
}

bool AST::PredicateNode::insert(AST::Node *term)
{
    return this->termlists->insert(term);
}

// 在 PredicateNode 类中修改现有的 clone() 方法：
AST::Node* AST::PredicateNode::clone() const {
    PredicateNode* newNode = new PredicateNode(this->name);
    if (this->termlists) {
        newNode->termlists = this->termlists->clone();
    }
    return newNode;
}

AST::PredicateNode::~PredicateNode()
{
    delete this->termlists;
    std::cout << "Predicate Node: " << this->Node::name << " Destroy" << std::endl;
}