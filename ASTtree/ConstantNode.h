#include "Node.h"
namespace AST{
    class ConstantNode : public Node {
public:
    std::string name;

    ConstantNode(const std::string& n) : name(n) {}

    NodeType getType() const override { return CONSTANT; }
};  
}