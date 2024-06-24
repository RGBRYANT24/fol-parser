bison -d fol.y
flex fol.l
g++ -std=c++11 lex.yy.c fol.tab.c -o fol
./fol