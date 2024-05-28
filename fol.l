%{
#include "fol.tab.h"
extern int yyerror(const char *msg);
%}

%%

[ \t\n]                ; // skip spaces, tabs, newlines
"&"                    { return AND; }
"|"                    { return OR; }
"->"                   { return IMPLY; }
"!"                    { return NOT; }
"forall"               { return FORALL; }
"exists"               { return EXISTS; }
"="                    { return EQ; }
"("                    { return LPAREN; }
")"                    { return RPAREN; }
","                    { return COMMA; }
[A-Z][A-Za-z0-9_]*     { yylval.str = strdup(yytext); return PREDICATE; }
[f-h][A-Za-z0-9_]*     { yylval.str = strdup(yytext); return FUNCTION; }  
[a-e][A-Za-z0-9_]*     { yylval.str = strdup(yytext); return CONSTANT; }
[i-z][A-Za-z0-9_]*     { yylval.str = strdup(yytext); return VARIABLE; }
.                      { yyerror("invalid character"); }  
        
%%

int yywrap() { return 1; }