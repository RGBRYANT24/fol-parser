%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./ASTtree/VariableNode.h"
#define YY_DECL extern "C" int yylex()

using AST::Node;
char* result = nullptr;
//std::shared_ptr<Node> root;
AST::Node* root = nullptr;
void yyerror(const char *s); 
int yylex(void);
%}

%union {
    char *string;
    int number;
}

%token <string> VARIABLE PREDICATE
%token <number> CONSTANT
%token FORALL EXISTS AND OR NOT IMPLIES IFF

%type <string> formula
%type <string> term_list
%type <string> term
%type <string> atomic_formula

%%

formula    : FORALL VARIABLE formula             
            {
                
                printf("Forall Formula: forall %s %s\n", $2, $3); char* temp = (char *)malloc(strlen($2) + strlen($3) + 8); sprintf(temp, "forall %s %s", $2, $3); result = temp; $$ = temp; printf("result now: %s\n", result);
            }
             | EXISTS VARIABLE formula            { printf("Exists Formula: exists %s %s\n", $2, $3); char* temp = (char *)malloc(strlen($2) + strlen($3) + 8); sprintf(temp, "exists %s %s", $2, $3); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | formula AND formula                { printf("And Formula: %s and %s\n", $1, $3); char* temp = (char *)malloc(strlen($1) + strlen($3) + 5); sprintf(temp, "%s and %s", $1, $3); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | formula OR formula                 { printf("Or Formula: %s or %s\n", $1, $3); char* temp = (char *)malloc(strlen($1) + strlen($3) + 4); sprintf(temp, "%s or %s", $1, $3); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | NOT formula                       { printf("Not Formula: not %s\n", $2); char* temp = (char *)malloc(strlen($2) + 5); sprintf(temp, "not %s", $2); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | formula IMPLIES formula           { printf("Implies Formula: %s implies %s\n", $1, $3); char* temp = (char *)malloc(strlen($1) + strlen($3) + 9); sprintf(temp, "%s implies %s", $1, $3); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | formula IFF formula               { printf("Iff Formula: %s iff %s\n", $1, $3); char* temp = (char *)malloc(strlen($1) + strlen($3) + 5); sprintf(temp, "%s iff %s", $1, $3); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | '(' formula ')'                   { printf("Parenthesized Formula: (%s)\n", $2); char* temp = (char *)malloc(strlen($2) + 3); sprintf(temp, "(%s)", $2); result = temp; $$ = temp; printf("result now: %s\n", result);}
             | atomic_formula                    { printf("Atomic Formula: %s\n", $1); char* temp = strdup($1); result = temp; $$ = temp; } 

atomic_formula  : PREDICATE '(' term_list ')'      { printf("Atomic Formula: %s(%s)\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 3); sprintf($$, "%s(%s)", $1, $3); }
                    | PREDICATE {printf("Atomic Formula: %s\n", $1);}

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