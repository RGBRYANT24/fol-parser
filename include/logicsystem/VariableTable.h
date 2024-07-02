#ifndef LOGIC_SYSTEM_VARIABLE_TABLE_H
#define LOGIC_SYSTEM_VARIABLE_TABLE_H

#include <vector>
#include <string>
#include <unordered_map>

namespace LogicSystem
{
    class VariableTable
    {
    private:
        std::vector<std::string> variables;
        std::unordered_map<std::string, int> variableToId;

    public:
        int insert(const std::string& variable);

        std::string get(int id) const;

        int getId(const std::string& variable) const;

        size_t size() const { return variables.size(); }
    };
}

#endif // LOGIC_SYSTEM_VARIABLE_TABLE_H