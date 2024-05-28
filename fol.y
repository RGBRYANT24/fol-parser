%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int yylex();
extern int yyparse();
extern FILE* yyin;

void yyerror(const char* s);
%}

%union {
	char *str;
}

%token AND OR IMPLY NOT FORALL EXISTS EQ 
%token LPAREN RPAREN COMMA
%token <str> PREDICATE FUNCTION CONSTANT VARIABLE

%type <str> formula atom term termlist

%%

formula: 
  atom                                 { $$ = $1; printf("Parsed formula: %s\n", $$); }  
| formula AND formula                  { $$ = $1; printf("Parsed formula: %s & %s\n", $1, $3); }
| formula OR formula                   { $$ = $1; printf("Parsed formula: %s | %s\n", $1, $3); }
| formula IMPLY formula                { $$ = $1; printf("Parsed formula: %s -> %s\n", $1, $3); }
| NOT formula                          { $$ = ""; printf("Parsed formula: !%s\n", $2); }        
| FORALL VARIABLE formula              { $$ = ""; printf("Parsed formula: forall %s %s\n", $2, $3); }
| EXISTS VARIABLE formula              { $$ = ""; printf("Parsed formula: exists %s %s\n", $2, $3); }
| LPAREN formula RPAREN                { $$ = $2; }
;

atom:  
  PREDICATE LPAREN termlist RPAREN     { $$ = $1; printf("Parsed atom: %s(%s)\n", $1, $3); }
| term EQ term                         { $$ = $1; printf("Parsed atom: %s=%s\n", $1, $3); }  
;
      
termlist:
  term                                 { $$ = $1; }  
| term COMMA termlist                  { $$ = $1; }
;
  
term:
  CONSTANT                             { $$ = $1; }  
| VARIABLE                             { $$ = $1; }
| FUNCTION LPAREN termlist RPAREN      { $$ = $1; printf("Parsed term: %s(%s)\n", $1, $3); }
;

%%

int main() {
  yyin = stdin;

  do { 
    yyparse();
  } while(!feof(yyin));

  return 0;
}

void yyerror(const char* s) {
  fprintf(stderr, "Parse error: %s\n", s);
  exit(1);
}