#include "FunctionNode.h"

AST::FunctionNode::FunctionNode()
{
    this -> name = "";
    this -> termlists = nullptr;
}

bool AST::FunctionNode::insert(AST::Node *term)
{
    return this -> termlists -> insert(term);
}

void AST::FunctionNode::print()
{
    std::cout << "Function Node print, Function Name: " << this->Node::name << std::endl;
    if(this -> termlists == nullptr){
        std::cout<<"No function term lists " << std::endl;
        return;
    }
    else{
        this -> termlists -> print();
    }
}

AST::FunctionNode::~FunctionNode()
{
    delete this -> termlists;
    std::cout << "Function Node: " << this->Node::name << " Destroy" << std::endl;
}

