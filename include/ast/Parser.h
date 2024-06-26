#ifndef PARSER_H
#define PARSER_H

#include <memory>
#include "AllNodes.h"

extern AST::Node* root;
#ifdef __cplusplus
extern "C" {
#endif
int yyparse(void);
#ifdef __cplusplus
}
#endif

#endif // PARSER_H