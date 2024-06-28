#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace AST
{
    class Node
    {
    protected:
        std::string content; //便于打印信息
    public:
        enum NodeType
        {
            PREDICATE,
            FUNCTION,
            VARIABLE,
            CONSTANT,
            TERMLIST,
            TERM,
            AND,
            OR,
            IMPLY,
            NOT,
            FORALL,
            EXISTS,
            EQ
        };
        std::string name;//节点名字 
        virtual NodeType getType() const = 0;
        virtual void print() {};
        virtual bool insert(Node* term) {return false;}; // insert arguments
        virtual ~Node() {} // Virtual destructor for proper cleanup
    };
}

#endif // NODE_H