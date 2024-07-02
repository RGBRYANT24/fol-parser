#include "VariableTable.h"

namespace LogicSystem
{
    int VariableTable::insert(const std::string& variable) {
        auto it = variableToId.find(variable);
        if (it != variableToId.end()) {
            return it->second;
        }
        int newId = variables.size();
        variables.push_back(variable);
        variableToId[variable] = newId;
        return newId;
    }

    std::string VariableTable::get(int id) const {
        if (id >= 0 && id < variables.size()) {
            return variables[id];
        }
        return "";
    }

    int VariableTable::getId(const std::string& variable) const {
        auto it = variableToId.find(variable);
        if (it != variableToId.end()) {
            return it->second;
        }
        return -1;
    }
}