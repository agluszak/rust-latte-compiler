// -- programs ------------------------------------------------
//
// entrypoints Program ;
//
// Program.   Program ::= [TopDef] ;
//
// FnDef.	   TopDef ::= Type Ident "(" [Arg] ")" Block ;
//
// separator nonempty TopDef "" ;
//
// Arg. 	   Arg ::= Type Ident;
//
// separator  Arg "," ;
//
// -- statements ----------------------------------------------
//
// Block.     Block ::= "{" [Stmt] "}" ;
//
// separator  Stmt "" ;
//
// Empty.     Stmt ::= ";" ;
//
// BStmt.     Stmt ::= Block ;
//
// Decl.      Stmt ::= Type [Item] ";" ;
//
// NoInit.    Item ::= Ident ;
//
// Init.      Item ::= Ident "=" Expr ;
//
// separator nonempty Item "," ;
//
// Ass.       Stmt ::= Ident "=" Expr  ";" ;
//
// Incr.      Stmt ::= Ident "++"  ";" ;
//
// Decr.      Stmt ::= Ident "--"  ";" ;
//
// Ret.       Stmt ::= "return" Expr ";" ;
//
// VRet.      Stmt ::= "return" ";" ;
//
// Cond.      Stmt ::= "if" "(" Expr ")" Stmt  ;
//
// CondElse.  Stmt ::= "if" "(" Expr ")" Stmt "else" Stmt  ;
//
// While.     Stmt ::= "while" "(" Expr ")" Stmt ;
//
// SExp.      Stmt ::= Expr  ";" ;
//
// -- Types ---------------------------------------------------
//
// Int.       Type ::= "int" ;
//
// Str.       Type ::= "string" ;
//
// Bool.      Type ::= "boolean" ;
//
// Void.      Type ::= "void" ;
//
// internal   Fun. Type ::= Type "(" [Type] ")" ;
//
// separator  Type "," ;
//
// -- Expressions ---------------------------------------------
//
// EVar.      Expr6 ::= Ident ;
//
// ELitInt.   Expr6 ::= Integer ;
//
// ELitTrue.  Expr6 ::= "true" ;
//
// ELitFalse. Expr6 ::= "false" ;
//
// EApp.      Expr6 ::= Ident "(" [Expr] ")" ;
//
// EString.   Expr6 ::= String ;
//
// Neg.       Expr5 ::= "-" Expr6 ;
//
// Not.       Expr5 ::= "!" Expr6 ;
//
// EMul.      Expr4 ::= Expr4 MulOp Expr5 ;
//
// EAdd.      Expr3 ::= Expr3 AddOp Expr4 ;
//
// ERel.      Expr2 ::= Expr2 RelOp Expr3 ;
//
// EAnd.      Expr1 ::= Expr2 "&&" Expr1 ;
//
// EOr.       Expr ::= Expr1 "||" Expr ;
//
// coercions  Expr 6 ;
//
// separator  Expr "," ;
//
// -- operators -----------------------------------------------
//
// Plus.      AddOp ::= "+" ;
//
// Minus.     AddOp ::= "-" ;
//
// Times.     MulOp ::= "*" ;
//
// Div.       MulOp ::= "/" ;
//
// Mod.       MulOp ::= "%" ;
//
// LTH.       RelOp ::= "<" ;
//
// LE.        RelOp ::= "<=" ;
//
// GTH.       RelOp ::= ">" ;
//
// GE.        RelOp ::= ">=" ;
//
// EQU.       RelOp ::= "==" ;
//
// NE.        RelOp ::= "!=" ;
//
// -- comments ------------------------------------------------
//
// comment    "#" ;
//
// comment    "//" ;
//
// comment    "/*" "*/" ;
