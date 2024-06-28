#include "Clause.h"
#include <iostream>

namespace LogicSystem {
    void Clause::addLiteral(CNF* cnf) {
        auto it = this->literals.find(cnf->getPredicateName());
        if (it != this->literals.end()) {
            delete it->second;  // 删除旧的 CNF 对象
        }
        this->literals[cnf->getPredicateName()] = cnf;
    }

    void Clause::removeLiteral(const std::string& predicateName) {
        auto it = this->literals.find(predicateName);
        if (it != this->literals.end()) {
            delete it->second;
            this->literals.erase(it);
        }
    }

    void Clause::print() const {
        bool first = true;
        std::cout << "Clause Print " << std::endl;
        for (const auto& pair : this->literals) {
            if (!first) std::cout << " OR ";
            pair.second->print();
            first = false;
        }
        std::cout << "Clause Print End-------------------" << std::endl;
    }

    std::vector<CNF*> Clause::getAllLiterals() const {
        std::vector<CNF*> allLiterals;
        allLiterals.reserve(this->literals.size());
        for (const auto& pair : this->literals) {
            allLiterals.push_back(pair.second);
        }
        return allLiterals;
    }

    CNF* Clause::getLiteral(const std::string& predicateName) const {
        auto it = this->literals.find(predicateName);
        if (it != this->literals.end()) {
            return it->second;
        }
        return nullptr;
    }

    bool Clause::containsPredicate(const std::string& predicateName) const {
        return this->literals.find(predicateName) != this->literals.end();
    }

    size_t Clause::size() const {
        return this->literals.size();
    }

    Clause::~Clause() {
        for (const auto& pair : this->literals) {
            delete pair.second;
        }
    }
}