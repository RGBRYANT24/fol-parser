#include "CNF.h"
#include <iostream>

namespace LogicSystem
{
    CNF::CNF(AST::Node *p)
        : predicate(p)
    {
        //如果不用not Predicate 或者 Predicate来初始化，直接报错并且终止
        if(p -> getType() != AST::Node::NOT && p -> getType() != AST::Node::PREDICATE)
        {
            std::cout << "construct cnf with wrong node, node : ";
            p -> print();
            delete p;
            exit(1);
        }
        if (p->getType() == AST::Node::NOT)
        {
            this->negated = true;
        }
        else
        {
            this->negated = false;
        }
    }

    void CNF::print() const
    {
        predicate->print();
    }

    std::string CNF::getPredicateName() const
    {
        return this->predicate->name;
    }

    bool CNF::isNegated() const
    {
        return this -> negated;
    }

    CNF::~CNF()
    {
        delete this -> predicate;
    }
}