#ifndef LOGIC_SYSTEM_CNF_H
#define LOGIC_SYSTEM_CNF_H

#include "AllNodes.h"
#include <string>

namespace LogicSystem
{
    class CNF
    {
    public:
        CNF(AST::Node *p);
        void print() const;
        std::string getPredicateName() const;
        bool isNegated() const;
        CNF(const CNF& other); //拷贝构造
        CNF& operator=(const CNF& other);  // 赋值运算符
        ~CNF();

    private:
        AST::Node *predicate;
        bool negated;
    };
}

#endif // LOGIC_SYSTEM_CNF_H