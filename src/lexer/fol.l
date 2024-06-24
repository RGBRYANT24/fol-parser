%{
#include <memory>
//#include "../../include/AllNodes.h"
#include "AllNodes.h"
#include "fol.tab.hh"  // Include Bison-generated header
%}

%%
"forall"                        { return FORALL; }
"exists"                        { return EXISTS; }
"and"                           { return AND; }
"or"                            { return OR; }
"not"                           { return NOT; }
"implies"                       { return IMPLIES; }
"iff"                           { return IFF; }
[f-h][A-Za-z0-9_]*              { yylval.string = strdup(yytext); return FUNCTION;}
[i-z][A-Za-z0-9_]*              { yylval.string = strdup(yytext); return VARIABLE; }
[A-Z][A-Za-z0-9_]*              { yylval.string = strdup(yytext); return PREDICATE; }
[0-9]+              { yylval.number = atoi(yytext); return CONSTANT; }

[()=,()]            { return yytext[0]; }
[ \t\n]             ; // Ignore whitespace
.                   { printf("Error: Unknown character '%c'\n", yytext[0]); }
%%

int yywrap(void) { return 1; } 
