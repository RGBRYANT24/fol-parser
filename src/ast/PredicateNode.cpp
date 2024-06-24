#include "PredicataNode.h"

void AST::PredicateNode::print(){
    std::cout << "PredicateNode name " << name << " arguments: ";
    for(const auto arg : this -> arguments){
        std::cout << arg << " ";
    }
    std::cout << std::endl;
}