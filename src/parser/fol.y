

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <iostream>
#include "AllNodes.h"
#define YY_DECL extern "C" int yylex()

using AST::Node;
char* result = nullptr;
AST::Node* root = nullptr;
void yyerror(const char *s); 
int yylex(void);

#ifdef __cplusplus
extern "C" {
#endif
int yyparse(void);
#ifdef __cplusplus
}
#endif


%}

%union {
    char *string;
    int number;
    AST::Node *node;
}

%token <string> VARIABLE PREDICATE FUNCTION
%token <number> CONSTANT
%token FORALL EXISTS AND OR NOT IMPLIES IFF

%type <string> formula
%type <node> term_list
%type <node> term
%type <node> non_function_term
%type <node> atomic_formula

%%

atomic_formula  : 
    PREDICATE '(' term_list ')'   
        {
            std::cout << "Atomic Formula with Term List" << std::endl;
            AST::PredicateNode* tmp = new AST::PredicateNode($1, $3);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }
    | PREDICATE
        {
            std::cout << "Atomic Formula (Predicate only)" << std::endl;
            AST::PredicateNode* tmp = new AST::PredicateNode($1);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }

term        : 
    term_list
        {
            $$ = $1;
            root = $$;
            std::cout<< "Term Node: ";
            $$ -> print();
        }
    | FUNCTION '(' term_list ')'
        {
            std::cout << "Function Node as Term" << std::endl;
            AST::FunctionNode* tmp = new AST::FunctionNode($1, $3);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }
    | FUNCTION '(', ')'
        {
            //function可以是零元函数
            std::cout << "Function Node Without params as Term" << std::endl;
            AST::FunctionNode* tmp = new AST::FunctionNode();
            tmp -> name = $1;
            $$ = tmp;
            root = tmp;
        }

term_list   :
    non_function_term
        {
            std::cout << "Term List Single Term " << std::endl;
            AST::TermListNode* tmp = new AST::TermListNode();
            tmp -> insert($1);
            root = tmp;
            $$ = tmp;
        }
    | term_list ',' non_function_term
        {
            std::cout << "Term List Multiple Terms " << std::endl;
            AST::TermListNode* tmp = dynamic_cast<AST::TermListNode*>($1);
            tmp -> insert($3);
            $$ = tmp;
        }

non_function_term :
    VARIABLE                           
        { 
            $$ = new AST::VariableNode($1);
            $$ -> print();
            printf("Term (Variable): %s\n", $1); 
        }
    | CONSTANT
        {
            $$ = new AST::ConstantNode(std::to_string($1));
            $$ -> print();
            std::cout << "Term (Constant): " << $1 << std::endl;
        }



%%



void yyerror(const char *s) {
    fprintf(stderr, "%s\n", s);
}