#ifndef LOGIC_SYSTEM_CLAUSE_H
#define LOGIC_SYSTEM_CLAUSE_H

#include <unordered_map>
#include <string>
#include <vector>
#include "CNF.h"

namespace LogicSystem {
    class Clause {
    public:
        void addLiteral(CNF* cnf);
        void removeLiteral(const std::string& predicateName);
        void print() const;
        std::vector<CNF*> getAllLiterals() const;
        CNF* getLiteral(const std::string& predicateName) const;
        bool containsPredicate(const std::string& predicateName) const;
        size_t size() const;
        ~Clause();

    private:
        std::unordered_map<std::string, CNF*> literals;
    };
}

#endif // LOGIC_SYSTEM_CLAUSE_H