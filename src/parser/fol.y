

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

%token <string> VARIABLE PREDICATE
%token <number> CONSTANT
%token FORALL EXISTS AND OR NOT IMPLIES IFF FUNCTION

%type <string> formula
%type <node> term_list
%type <node> term
%type <node> atomic_formula

%%



term_list   :
    term
        {
            std::cout << "Term List Term " << std::endl;
            AST::TermListNode* tmp = new AST::TermListNode();
            tmp -> insert($1);
            root = tmp;
            $$ = tmp;
        }
    | term_list ',' term
        {
            std::cout << "Term List Lists " << std::endl;
            AST::TermListNode* tmp = dynamic_cast<AST::TermListNode*>($1);
            tmp -> insert($3);
            root = tmp;
            $$ = tmp;
        }

term        : 
    VARIABLE                           
        { 
            printf("Term (Variable): %s\n", $1); 
            $$ = new AST::VariableNode($1);
            $$ -> print();
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