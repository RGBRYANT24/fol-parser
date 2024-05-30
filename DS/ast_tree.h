#include <string>
#include <vector>
#include <memory>
#include <iostream>

class Node {
public:
    enum NodeType {
        PREDICATE, FUNCTION, VARIABLE, CONSTANT,
        AND, OR, IMPLY, NOT, FORALL, EXISTS, EQ
    };

    virtual NodeType getType() const = 0;
    virtual ~Node() {} // Virtual destructor for proper cleanup
};

// Leaf Nodes

class PredicateNode : public Node {
public:
    std::string name;
    std::vector<std::shared_ptr<Node>> arguments; // Arguments could be vars, constants, functions

    PredicateNode(const std::string& n) : name(n) {}

    NodeType getType() const override { return PREDICATE; }
};

class FunctionNode : public Node {
public:
    std::string name;
    std::vector<std::shared_ptr<Node>> arguments; 

    FunctionNode(const std::string& n) : name(n) {}

    NodeType getType() const override { return FUNCTION; }
};

class VariableNode : public Node {
public:
    std::string name;

    VariableNode(const std::string& n) : name(n) {}

    NodeType getType() const override { return VARIABLE; }
};

class ConstantNode : public Node {
public:
    std::string name;

    ConstantNode(const std::string& n) : name(n) {}

    NodeType getType() const override { return CONSTANT; }
};


// Non-Leaf Nodes (Internal Nodes)

class AndNode : public Node {
public:
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    AndNode(std::shared_ptr<Node> l, std::shared_ptr<Node> r) : left(l), right(r) {}

    NodeType getType() const override { return AND; }
};

// ... (similar classes for OR, IMPLY, NOT, FORALL, EXISTS, EQ)
class OrNode : public Node {
public:
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    OrNode(std::shared_ptr<Node> l, std::shared_ptr<Node> r) : left(l), right(r) {}

    NodeType getType() const override { return OR; }
};

class ImplyNode : public Node {
public:
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    ImplyNode(std::shared_ptr<Node> l, std::shared_ptr<Node> r) : left(l), right(r) {}

    NodeType getType() const override { return IMPLY; }
};

class NotNode : public Node {
public:
    std::shared_ptr<Node> operand;

    NotNode(std::shared_ptr<Node> op) : operand(op) {}

    NodeType getType() const override { return NOT; }
};

class ForallNode : public Node {
public:
    std::shared_ptr<Node> variable;
    std::shared_ptr<Node> formula;

    ForallNode(std::shared_ptr<Node> var, std::shared_ptr<Node> form) : variable(var), formula(form) {}

    NodeType getType() const override { return FORALL; }
};

class ExistsNode : public Node {
public:
    std::shared_ptr<Node> variable;
    std::shared_ptr<Node> formula;

    ExistsNode(std::shared_ptr<Node> var, std::shared_ptr<Node> form) : variable(var), formula(form) {}

    NodeType getType() const override { return EXISTS; }
};

class EqNode : public Node {
public:
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    EqNode(std::shared_ptr<Node> l, std::shared_ptr<Node> r) : left(l), right(r) {}

    NodeType getType() const override { return EQ; }
};
