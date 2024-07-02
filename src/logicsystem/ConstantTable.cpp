#include "ConstantTable.h"

namespace LogicSystem
{
    int ConstantTable::insert(const std::string& constant) {
        auto it = constantToId.find(constant);
        if (it != constantToId.end()) {
            return it->second;
        }
        int newId = constants.size();
        constants.push_back(constant);
        constantToId[constant] = newId;
        return newId;
    }

    std::string ConstantTable::get(int id) const {
        if (id >= 0 && id < constants.size()) {
            return constants[id];
        }
        return "";
    }

    int ConstantTable::getId(const std::string& constant) const {
        auto it = constantToId.find(constant);
        if (it != constantToId.end()) {
            return it->second;
        }
        return -1;
    }
}