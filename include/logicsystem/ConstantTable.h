#ifndef LOGIC_SYSTEM_CONSTANT_TABLE_H
#define LOGIC_SYSTEM_CONSTANT_TABLE_H

#include <vector>
#include <string>
#include <unordered_map>

namespace LogicSystem
{
    class ConstantTable
    {
    private:
        std::vector<std::string> constants;
        std::unordered_map<std::string, int> constantToId;

    public:
        int insert(const std::string& constant);

        std::string get(int id) const;

        int getId(const std::string& constant) const;

        size_t size() const { return constants.size(); }
    };
}

#endif // LOGIC_SYSTEM_CONSTANT_TABLE_H