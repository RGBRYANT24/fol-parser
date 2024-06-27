

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

%type <node> formula
%type <node> term_list
%type <node> term
%type <node> non_function_term function_term
%type <node> atomic_formula

%%

formula :
    NOT formula
    {
        std::cout << "Not formula " << std::endl;
        AST::UnaryOpNode* tmp = new AST::UnaryOpNode(AST::Node::NodeType::NOT, $2);
        root = tmp;
        $$ = root;
    }
    |'(' formula ')'
    {
            $$ = $2;
    }
    |atomic_formula
    {
        std::cout << "Formula construct by atomic_formula " << std::endl;
        $$ = $1;
    }
    | term
    {
        std::cout << "Formula construct by term" << std::endl;
        $$ = $1;
    }

atomic_formula  : 
    PREDICATE '(' term ')'   
        {
            std::cout << "Atomic Formula with Term" << std::endl;
            AST::PredicateNode* tmp = new AST::PredicateNode($1, $3);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }
    | PREDICATE '(' term_list ')'
        {
            std::cout << "Atomic Formula with Term Lists" << std::endl;
            AST::PredicateNode* tmp = new AST::PredicateNode($1, $3);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }
    | PREDICATE '(' ')'
        {
            std::cout << "Atomic Formula (Predicate only)" << std::endl;
            AST::PredicateNode* tmp = new AST::PredicateNode($1);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }

term        : 
    non_function_term
        {
            $$ = $1;
            root = $$;
            std::cout<< "Term Node: ";
            $$ -> print();
        }
    | function_term
        {
            $$ = $1;
            root = $$;
            std::cout<< "Term Node with function_term: ";
            $$ -> print();
        }

function_term  :
    FUNCTION '(' ')'
        {
            //function可以是零元函数
            std::cout << "Function Node Without params as Term" << std::endl;
            AST::FunctionNode* tmp = new AST::FunctionNode();
            tmp -> name = $1;
            $$ = tmp;
            root = tmp;
        }
    | FUNCTION '(' term_list ')'
        {
            std::cout << "Function Node as Term" << std::endl;
            AST::FunctionNode* tmp = new AST::FunctionNode($1, $3);
            root = tmp;
            $$ = tmp;
            $$ -> print();
        }
    

term_list   :
    term
        {
            std::cout << "Term List Single Term " << std::endl;
            AST::TermListNode* tmp = new AST::TermListNode();
            tmp -> insert($1);
            root = tmp;
            $$ = tmp;
        }
    | term ',' term_list
        {
            std::cout << "Term List Multiple Terms " << std::endl;
            AST::TermListNode* tmp = dynamic_cast<AST::TermListNode*>($3);
            tmp -> insert($1);
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