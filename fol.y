// Header and Declarations
%{

// In your fol.ypp file (or a separate header file)
#define YYSTYPE std::string

#include <memory>
#include "DS/ast_tree.h" // Include your node classes

extern int yylex();
extern int yyparse();
%}
%define api.value.type {char*}
// Token types from flex
%token AND OR IMPLY NOT FORALL EXISTS EQ LPAREN RPAREN COMMA PREDICATE FUNCTION CONSTANT VARIABLE

// Associativity and Precedence
%left IMPLY 
%left OR
%left AND
%right NOT
%left EQ

// Grammar Rules and AST Construction
%%

formula:
    FORALL VARIABLE formula                { $$ = std::make_shared<ForallNode>(std::make_shared<VariableNode>($2), $3); }
    | EXISTS VARIABLE formula                { $$ = std::make_shared<ExistsNode>(std::make_shared<VariableNode>($2), $3); }
    | formula IMPLY formula               { $$ = std::make_shared<ImplyNode>($1, $3); }
    | formula OR formula                  { $$ = std::make_shared<OrNode>($1, $3); }
    | formula AND formula                 { $$ = std::make_shared<AndNode>($1, $3); }
    | NOT formula                         { $$ = std::make_shared<NotNode>($2); }
    | term EQ term                       { $$ = std::make_shared<EqNode>($1, $3); }
    | LPAREN formula RPAREN                { $$ = $2; }  // Eliminate parentheses
    | predicate                            
    | function
    ;

term:
    CONSTANT                               { $$ = std::make_shared<ConstantNode>($1); }
    | VARIABLE                               { $$ = std::make_shared<VariableNode>($1); }
    | function                                
    ;

predicate:
    PREDICATE                              { $$ = std::make_shared<PredicateNode>($1); }
    | PREDICATE LPAREN args RPAREN           { $$ = std::make_shared<PredicateNode>($1); $$->arguments = $3; }  // Store arguments
    ;

function:
    FUNCTION                               { $$ = std::make_shared<FunctionNode>($1); }
    | FUNCTION LPAREN args RPAREN           { $$ = std::make_shared<FunctionNode>($1); $$->arguments = $3; } 
    ;

args:
    term                                   { $$ = std::vector<std::shared_ptr<Node>>{ $1 }; }
    | args COMMA term                       { $1.push_back($3); $$ = $1; }
    ;
%%
