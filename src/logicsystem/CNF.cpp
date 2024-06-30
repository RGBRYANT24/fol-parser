#include "CNF.h"
#include <iostream>

namespace LogicSystem
{
    CNF::CNF(AST::Node *p)
    {
        if (p)
        {
            // 如果不用not Predicate 或者 Predicate来初始化，直接报错并且终止
            if (p->getType() != AST::Node::NOT && p->getType() != AST::Node::PREDICATE)
            {
                std::cout << "construct cnf with wrong node, node : ";
                p->print();
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
            this->predicate = p->clone(); // 深拷贝
        }
        else
        {
            std::cout << "construct cnf with nullptr node" << std::endl;
        }
    }

    void CNF::print() const
    {
        if (this->negated)
        {
            std::cout << "Not node ";
        }
        predicate->print();
    }

    std::string CNF::getPredicateName() const
    {
        return this->predicate->name;
    }

    bool CNF::isNegated() const
    {
        return this->negated;
    }

    // 拷贝构造函数
    CNF::CNF(const CNF &other)
    {
        this->negated = other.negated;
        if (other.predicate)
        {
            this->predicate = other.predicate->clone(); // 使用clone方法进行深拷贝
        }
        else
        {
            this->predicate = nullptr;
        }
    }

    // 赋值运算符
    CNF &CNF::operator=(const CNF &other)
    {
        if (this != &other) // 自赋值检查
        {
            delete this->predicate; // 删除原有的predicate

            this->negated = other.negated;
            if (other.predicate)
            {
                this->predicate = other.predicate->clone(); // 使用clone方法进行深拷贝
            }
            else
            {
                this->predicate = nullptr;
            }
        }
        return *this;
    }

    CNF::~CNF()
    {
        delete this->predicate;
    }
}