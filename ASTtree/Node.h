#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace AST
{
    class Node
    {
    protected:
        std::string content;//便于打印信息
    public:
        enum NodeType
        {
            PREDICATE,
            FUNCTION,
            VARIABLE,
            CONSTANT,
            AND,
            OR,
            IMPLY,
            NOT,
            FORALL,
            EXISTS,
            EQ
        };
        virtual NodeType getType() const = 0;
        virtual void print() {};
        virtual ~Node() {} // Virtual destructor for proper cleanup

    };
}