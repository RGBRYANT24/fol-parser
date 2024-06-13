#include "Node.h"

namespace AST
{
    class VariableNode : public Node
    {
    public:
        std::string name;

        VariableNode(const std::string &n) : name(n) {}
        inline void print() {std::cout<<"Variable Node " << this -> name << std::endl;}
        NodeType getType() const override { return VARIABLE; }
    };
}