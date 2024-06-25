#include "PredicataNode.h"

AST::PredicateNode::PredicateNode()
{
    this -> name = "";
    this -> termlists = nullptr;
}

void AST::PredicateNode::print(){
    std::cout << "PredicateNode name " << name << " arguments: ";
    if(this -> termlists == nullptr){
        std::cout<<"No Predicate term lists " << std::endl;
        return;
    }
    else{
        this -> termlists -> print();
    }
    std::cout << std::endl;
}

bool AST::PredicateNode::insert(AST::Node *term)
{
    return this -> termlists -> insert(term);
}

AST::PredicateNode::~PredicateNode()
{
    delete this -> termlists;
    std::cout << "Predicate Node: " << this->name << " Destroy" << std::endl;
}