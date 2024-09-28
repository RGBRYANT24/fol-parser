#ifndef LOGIC_SYSTEM_PREDICATE_TABLE_H
#define LOGIC_SYSTEM_PREDICATE_TABLE_H
#include <vector>
#include <string>
#include <unordered_map>


namespace LogicSystem
{
    class PredicateTable
    {
    private:
        std::vector<std::string> predicates;
        std::unordered_map<std::string, int> predicateToId;

    public:
        int insert(const std::string &predicate);

        std::string get(int id) const;

        int getId(const std::string &predicate) const;

        size_t size() const { return predicates.size(); }
    };
}

#endif