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
    //testNode node;
}

%token <string> VARIABLE PREDICATE
%token <number> CONSTANT
%token FORALL EXISTS AND OR NOT IMPLIES IFF

%type <string> formula
%type <string> term_list
%type <string> term
%type <string> atomic_formula

%%


atomic_formula  : PREDICATE '(' term_list ')'      
                { 
                    printf("Atomic Formula: %s(%s)\n", $1, $3); 
                    $$ = (char *)malloc(strlen($1) + strlen($3) + 3); 
                    sprintf($$, "%s(%s)", $1, $3); 
                }
                | PREDICATE 
                {
                    printf("Atomic Formula: %s\n", $1);

                }

term_list   : term                             { printf("Term List: %s\n", $1); $$ = strdup($1); }
             | term_list ',' term                { printf("Term List: %s, %s\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 2); sprintf($$, "%s, %s", $1, $3); }

term        : VARIABLE                           { 
    printf("Creating Variable AST NODE\n");
    printf("Term (Variable): %s\n", $1); 
    $$ = strdup($1); 
    std::shared_ptr<Node> tmp = std::make_shared<AST::VariableNode>($$);
    //AST::VariableNode* tmp = new AST::VariableNode($$);
    tmp -> print();
    //delete tmp;
    }
             | CONSTANT                          { printf("Term (Constant): %d\n", $1); $$ = (char *)malloc(12); sprintf($$, "%d", $1); }

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