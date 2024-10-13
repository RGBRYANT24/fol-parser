#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <sstream>
#include <memory>
#include <ctime>
struct Clause {
    std::vector<int> literals;
    int id;
    std::shared_ptr<Clause> father;
    std::shared_ptr<Clause> mother;

    Clause(const std::vector<int>& lits, int _id = -1)
            : literals(lits), id(_id), father(nullptr), mother(nullptr) {
        std::sort(literals.begin(), literals.end());
    }

    bool operator==(const Clause& other) const {
        return literals == other.literals;
    }

    Clause resolve(const Clause& other, int lit) const {
        std::vector<int> newLiterals;
        for (int l : literals) {
            if (l != lit) newLiterals.push_back(l);
        }
        for (int l : other.literals) {
            if (l != -lit && std::find(newLiterals.begin(), newLiterals.end(), l) == newLiterals.end()) {
                newLiterals.push_back(l);
            }
        }
        return Clause(newLiterals);
    }

    bool isEmpty() const {
        return literals.empty();
    }

    bool isTautology() const {
        for (size_t i = 0; i < literals.size(); ++i) {
            for (size_t j = i + 1; j < literals.size(); ++j) {
                if (literals[i] == -literals[j]) return true;
            }
        }
        return false;
    }

    friend std::ostream& operator<<(std::ostream& os, const Clause& clause) {
        os << "{";
        for (size_t i = 0; i < clause.literals.size(); ++i) {
            os << clause.literals[i];
            if (i < clause.literals.size() - 1) os << ", ";
        }
        os << "}";
        return os;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

struct KnowledgeBase {
    std::vector<std::shared_ptr<Clause>> clauses;
    int nextId = 0;

    void addClause(const Clause& clause) {
        if (!contains(clause)) {
            auto newClause = std::make_shared<Clause>(clause.literals, nextId++);
            newClause->father = std::make_shared<Clause>(std::vector<int>{}, -1); // root
            newClause->mother = std::make_shared<Clause>(std::vector<int>{}, -1); // root
            clauses.push_back(newClause);
        }
    }

    bool contains(const Clause& clause) const {
        return std::find_if(clauses.begin(), clauses.end(),
                            [&](const std::shared_ptr<Clause>& c) { return *c == clause; }) != clauses.end();
    }

    friend std::ostream& operator<<(std::ostream& os, const KnowledgeBase& kb) {
        os << "Knowledge Base:\n";
        for (const auto& clause : kb.clauses) {
            os << "  " << *clause << " (ID: " << clause->id << ")\n";
        }
        return os;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

struct ClauseComparator {
    bool operator()(const std::shared_ptr<Clause>& a, const std::shared_ptr<Clause>& b) {
        return a->literals.size() > b->literals.size();
    }
};

std::vector<std::pair<int, int>> getResolvableClauses(const Clause& clause, const KnowledgeBase& kb) {
    std::vector<std::pair<int, int>> resolvableClauses;
    for (size_t i = 0; i < kb.clauses.size(); ++i) {
        for (int lit : clause.literals) {
            if (std::binary_search(kb.clauses[i]->literals.begin(), kb.clauses[i]->literals.end(), -lit)) {
                resolvableClauses.emplace_back(i, lit);
                break;
            }
        }
    }
    return resolvableClauses;
}

void printAncestry(const std::shared_ptr<Clause>& clause, std::ofstream& outFile, int depth = 0) {
    std::string indent(depth * 2, ' ');
    outFile << indent << "Clause: " << *clause << " (ID: " << clause->id << ")\n";

    if (clause->father->id == -1 && clause->mother->id == -1) {
        outFile << indent << "  This is a root clause\n";
        return;
    }

    if (clause->father->id != -1) {
        outFile << indent << "  Father:\n";
        printAncestry(clause->father, outFile, depth + 1);
    }

    if (clause->mother->id != -1) {
        outFile << indent << "  Mother:\n";
        printAncestry(clause->mother, outFile, depth + 1);
    }
}

bool resolution(KnowledgeBase& kb, int& resolutionSteps, int maxSteps, std::ofstream& outFile) {
    auto print = [&outFile](const std::string& message) {
        std::cout << message;
        outFile << message;
    };

    print("Initial " + kb.toString() + "\n");

    std::priority_queue<std::shared_ptr<Clause>, std::vector<std::shared_ptr<Clause>>, ClauseComparator> pendingClauses;
    std::unordered_set<std::string> seenClauses;

    auto startClause = std::make_shared<Clause>(std::vector<int>{1, 2, 3}, kb.nextId++);
    startClause->father = std::make_shared<Clause>(std::vector<int>{}, -1); // root
    startClause->mother = std::make_shared<Clause>(std::vector<int>{}, -1); // root
    pendingClauses.push(startClause);

    for (const auto& clause : kb.clauses) {
        std::string clauseKey;
        for (int l : clause->literals) clauseKey += std::to_string(l) + ",";
        seenClauses.insert(clauseKey);
    }

    int round = 0;
    int maxLength = 3;

    while (resolutionSteps < maxSteps && !pendingClauses.empty()) {
        round++;
        print("\n--- Round " + std::to_string(round) + " ---\n");
        print("Current maxLength: " + std::to_string(maxLength) + "\n");
        print("Clauses for this round:\n");

        std::vector<std::shared_ptr<Clause>> currentRoundClauses;
        std::vector<std::shared_ptr<Clause>> postponedClauses;

        while (!pendingClauses.empty()) {
            auto clause = pendingClauses.top();
            pendingClauses.pop();

            if (clause->literals.size() <= maxLength) {
                currentRoundClauses.push_back(clause);
            } else {
                postponedClauses.push_back(clause);
            }

            if (currentRoundClauses.size() >= 10) break;
        }

        for (const auto& clause : currentRoundClauses) {
            print("  " + clause->toString() + "\n");
        }

        std::vector<std::shared_ptr<Clause>> newClauses;
        for (const auto& currentClause : currentRoundClauses) {
            print("\nCurrent clause: " + currentClause->toString() + "\n");

            kb.clauses.push_back(currentClause);
            print("Adding current clause to KB: " + currentClause->toString() + "\n");

            auto resolvableClauses = getResolvableClauses(*currentClause, kb);

            for (const auto& [clauseIndex, literal] : resolvableClauses) {
                auto resolvedClause = std::make_shared<Clause>(currentClause->resolve(*kb.clauses[clauseIndex], literal));
                resolvedClause->id = kb.nextId++;
                resolvedClause->father = currentClause;
                resolvedClause->mother = kb.clauses[clauseIndex];
                resolutionSteps++;

                print("Resolution step " + std::to_string(resolutionSteps) + ":\n");
                print("Resolving " + currentClause->toString() + " and " + kb.clauses[clauseIndex]->toString() + "\n");
                print("Resolved clause: " + resolvedClause->toString() + "\n");

                if (resolvedClause->isEmpty()) {
                    print("\nEmpty clause derived. UNSAT.\n");
                    print("\nAncestry of the empty clause:\n");
                    printAncestry(resolvedClause, outFile);
                    return true; // UNSAT
                }

                if (resolvedClause->isTautology()) {
                    print("Resolved clause is a tautology, skipping\n");
                    continue;
                }

                std::string clauseKey;
                for (int l : resolvedClause->literals) clauseKey += std::to_string(l) + ",";

                if (seenClauses.find(clauseKey) == seenClauses.end()) {
                    seenClauses.insert(clauseKey);
                    newClauses.push_back(resolvedClause);
                    print("New clause added to pending queue: " + resolvedClause->toString() + "\n");
                } else {
                    print("Clause already seen, skipping: " + resolvedClause->toString() + "\n");
                }
            }
        }

        for (const auto& newClause : newClauses) {
            pendingClauses.push(newClause);
        }
        for (const auto& postponedClause : postponedClauses) {
            pendingClauses.push(postponedClause);
        }

        if (newClauses.empty() && currentRoundClauses.empty()) {
            print("No new clauses generated in this round.\n");
            maxLength++;
            print("Increasing maxLength to: " + std::to_string(maxLength) + "\n");
        }
    }

    if (resolutionSteps >= maxSteps) {
        print("\nMax steps reached. Inconclusive.\n");
    } else {
        print("\nNo more clauses to resolve. SAT.\n");
    }

    return false;
}

int main() {
    clock_t start = clock();
    std::ofstream outFile("resolution.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening file resolution.txt" << std::endl;
        return 1;
    }

    KnowledgeBase kb;
    //初始化kb
    kb.addClause(Clause({1, 2, 3}));
    kb.addClause(Clause({4, 5, 6}));
    kb.addClause(Clause({7, 8, 9}));
    kb.addClause(Clause({10, 11, 12}));
    kb.addClause(Clause({-1, -4}));
    kb.addClause(Clause({-1, -7}));
    kb.addClause(Clause({-1, -10}));
    kb.addClause(Clause({-2, -5}));
    kb.addClause(Clause({-2, -8}));
    kb.addClause(Clause({-2, -11}));
    kb.addClause(Clause({-3, -6}));
    kb.addClause(Clause({-3, -9}));
    kb.addClause(Clause({-3, -12}));
    kb.addClause(Clause({-7, -10}));
    kb.addClause(Clause({-4, -7}));
    kb.addClause(Clause({-4, -10}));
    kb.addClause(Clause({-8, -11}));
    kb.addClause(Clause({-5, -8}));
    kb.addClause(Clause({-5, -11}));
    kb.addClause(Clause({-9, -12}));
    kb.addClause(Clause({-6, -9}));
    kb.addClause(Clause({-6, -12}));

    int resolutionSteps = 0;
    int maxSteps = 1000000;

    bool result = resolution(kb, resolutionSteps, maxSteps, outFile);

    std::string finalResult = "Resolution result: " + std::string(result ? "UNSAT" : "SAT") + "\n";
    finalResult += "Total resolution steps: " + std::to_string(resolutionSteps) + "\n";
    finalResult += "Final Knowledge Base:\n" + kb.toString() + "\n";
    std::cout << finalResult;
    outFile << finalResult;
    outFile.close();
    clock_t end = clock();

    double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    std::cout << "Runtime: " << time << " s" << std::endl;
    return 0;
}