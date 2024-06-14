%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include "./ASTtree/AllNodes.h"
#include "./ASTtree/Node.h"
#include "./ASTtree/VariableNode.h"
#define YY_DECL extern "C" int yylex()

using AST::Node;
char* result = nullptr;
std::shared_ptr<AST::Node> root;
//AST::Node* root = nullptr;
void yyerror(const char *s); 
int yylex(void);



%}

%union {
    char *string;
    int number;
    AST::Node *node;
}

%token <string> VARIABLE PREDICATE
%token <number> CONSTANT
%token FORALL EXISTS AND OR NOT IMPLIES IFF

%type <string> formula
%type <string> term_list
%type <node> term
%type <node> atomic_formula

%%




term        : 
    VARIABLE                           
        { 
            printf("Term (Variable): %s\n", $1); 
            $$ = new AST::VariableNode($1);
            $$ -> print();
        }
    | CONSTANT
        {
            // Convert integer to string
            //$$ = std::make_shared<AST::ConstantNode>(std::to_string($1));
            $$ = new AST::ConstantNode(std::to_string($1));
            $$ -> print();
            std::cout << "Term (Constant): " << $1 << std::endl;
        }

%%

int main(int argc, char *argv[]) {
    yyparse();
    printf("Result in main: %s\n", result);
    free(result); // 释放动态分配的内存
    return 0;
}

void yyerror(const char *s) {
    fprintf(stderr, "%s\n", s);
}