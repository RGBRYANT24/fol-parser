%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void yyerror(char *s); 
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
%type <string> term_list // Corrected type declaration for term_list
%type <string> term
%type <string> atomic_formula

%%

formula    : FORALL VARIABLE formula             { printf("Forall Formula: forall %s %s\n", $2, $3); $$ = (char *)malloc(strlen($2) + strlen($3) + 8); sprintf($$, "forall %s %s", $2, $3); }
             | EXISTS VARIABLE formula            { printf("Exists Formula: exists %s %s\n", $2, $3); $$ = (char *)malloc(strlen($2) + strlen($3) + 8); sprintf($$, "exists %s %s", $2, $3); }
             | formula AND formula                { printf("And Formula: %s and %s\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 5); sprintf($$, "%s and %s", $1, $3); }
             | formula OR formula                 { printf("Or Formula: %s or %s\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 4); sprintf($$, "%s or %s", $1, $3); }
             | NOT formula                       { printf("Not Formula: not %s\n", $2); $$ = (char *)malloc(strlen($2) + 5); sprintf($$, "not %s", $2); }
             | formula IMPLIES formula           { printf("Implies Formula: %s implies %s\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 9); sprintf($$, "%s implies %s", $1, $3); }
             | formula IFF formula               { printf("Iff Formula: %s iff %s\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 5); sprintf($$, "%s iff %s", $1, $3); }
             | '(' formula ')'                   { printf("Parenthesized Formula: (%s)\n", $2); $$ = (char *)malloc(strlen($2) + 3); sprintf($$, "(%s)", $2); }
             | atomic_formula                    { printf("Atomic Formula: %s\n", $1); $$ = strdup($1); } 

atomic_formula  : PREDICATE '(' term_list ')'      { printf("Atomic Formula: %s(%s)\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 3); sprintf($$, "%s(%s)", $1, $3); }

term_list   : term                             { printf("Term List: %s\n", $1); $$ = strdup($1); }
             | term_list ',' term                { printf("Term List: %s, %s\n", $1, $3); $$ = (char *)malloc(strlen($1) + strlen($3) + 2); sprintf($$, "%s, %s", $1, $3); }

term        : VARIABLE                           { printf("Term (Variable): %s\n", $1); $$ = strdup($1); }
             | CONSTANT                          { printf("Term (Constant): %d\n", $1); $$ = (char *)malloc(12); sprintf($$, "%d", $1); }

%%

int main(int argc, char *argv[]) { // Correct main function definition
    yyparse();
    return 0;
}

void yyerror(char *s) {
    fprintf(stderr, "%s\n", s);
}
