#include "PredicateTable.h"

namespace LogicSystem
{
    int PredicateTable::insert(const std::string &predicate)
        {
            auto it = predicateToId.find(predicate);
            if (it != predicateToId.end())
            {
                return it->second;
            }
            int newId = predicates.size();
            predicates.push_back(predicate);
            predicateToId[predicate] = newId;
            return newId;
        }

    std::string PredicateTable::get(int id) const
        {
            if (id >= 0 && id < predicates.size())
            {
                return predicates[id];
            }
            return "";
        }

    int PredicateTable::getId(const std::string &predicate) const
        {
            auto it = predicateToId.find(predicate);
            if (it != predicateToId.end())
            {
                return it->second;
            }
            return -1;
        }
}