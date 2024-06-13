%{
#include <memory>
#include "./ASTtree/AllNodes.h"
#include "fol.tab.h"  // Include Bison-generated header
%}

%%
[a-z]             { yylval.string = strdup(yytext); return VARIABLE; }
[A-Z]             { yylval.string = strdup(yytext); return PREDICATE; }
[0-9]+            { yylval.number = atoi(yytext); return CONSTANT; }
"forall"          { return FORALL; }
"exists"          { return EXISTS; }
"and"             { return AND; }
"or"              { return OR; }
"not"             { return NOT; }
"implies"         { return IMPLIES; }
"iff"             { return IFF; }
[()=,()]         { return yytext[0]; }
[ \t\n]           ; // Ignore whitespace
.                 { printf("Error: Unknown character '%c'\n", yytext[0]); }
%%

int yywrap(void) { return 1; } 
