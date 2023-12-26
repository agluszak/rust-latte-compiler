// 
// #[cfg(test)]
// mod test {
//     use crate::ir::*;
// 
//     mod ast {
//         use std::collections::BTreeMap;
//         use std::mem;
//         use crate::ir::test::{tac};
//         use crate::ir::{IrContext, VariableId};
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         enum Op {
//             Add,
//             Mod,
//             Lt,
//             Eq,
//         }
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         enum Expr {
//             Var(String),
//             Literal(i32),
//             Binary(Op, Box<Expr>, Box<Expr>),
//         }
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         struct BlockId(u32);
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         enum Stmt {
//             Assign(String, Expr),
//             If(Expr, Vec<Stmt>, Option<Vec<Stmt>>),
//             While(Expr, Vec<Stmt>),
//             Return(Expr),
//         }
// 
//         struct Context {
//             variables: BTreeMap<String, VariableId>,
//             next_variable_id: u32,
//             current_stmts: Vec<crate::ir::test::tac::Stmt>
//         }
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         pub(crate) enum Val {
//             Var(VariableId),
//             Literal(i32),
//         }
// 
//         impl Context {
//             fn new() -> Context {
//                 Context {
//                     variables: BTreeMap::new(),
//                     next_variable_id: 0,
//                     current_stmts: Vec::new()
//                 }
//             }
// 
//             fn child(&mut self) -> Context {
//                 Context {
//                     variables: self.variables.clone(),
//                     next_variable_id: self.next_variable_id,
//                     current_stmts: Vec::new()
//                 }
//             }
// 
//             fn new_variable(&mut self) -> VariableId {
//                 let id = VariableId::new(self.next_variable_id);
//                 self.next_variable_id += 1;
//                 id
//             }
// 
//             fn get_variable(&mut self, name: &str) -> VariableId {
//                 if let Some(id) = self.variables.get(name) {
//                     *id
//                 } else {
//                     let id = self.new_variable();
//                     self.variables.insert(name.to_string(), id);
//                     id
//                 }
//             }
// 
//             fn finish_block(&mut self, parent: &mut Self) -> Vec<crate::ir::test::tac::Stmt> {
//                 parent.next_variable_id = self.next_variable_id;
//                 std::mem::take(&mut self.current_stmts)
//             }
// 
//             fn emit(&mut self, stmt: crate::ir::test::tac::Stmt) {
//                 self.current_stmts.push(stmt);
//             }
// 
//             fn translate_expr(&mut self, expr: Expr) -> Val {
//                 match expr {
//                     Expr::Var(name) => {
//                         Val::Var(self.get_variable(&name))
//                     },
//                     Expr::Literal(lit) => {
//                         Val::Literal(lit)
//                     }
//                     Expr::Binary(op, lhs, rhs) => {
//                         let lhs = self.translate_expr(*lhs);
//                         let rhs = self.translate_expr(*rhs);
// 
//                         if let (Val::Literal(lhs), Val::Literal(rhs)) = (&lhs, &rhs) {
//                             let result = match op {
//                                 Op::Add => lhs + rhs,
//                                 Op::Mod => lhs % rhs,
//                                 Op::Lt => (lhs < rhs) as i32,
//                                 Op::Eq => (lhs == rhs) as i32,
//                             };
//                             return Val::Literal(result);
//                         }
// 
//                         let variable_id = self.new_variable();
//                         match op {
//                             Op::Add =>
//                                 self.emit(crate::ir::test::tac::Stmt::Assign(variable_id, crate::ir::test::tac::Op::Add(lhs, rhs))),
//                             Op::Mod =>
//                                 self.emit(crate::ir::test::tac::Stmt::Assign(variable_id, crate::ir::test::tac::Op::Mod(lhs, rhs))),
//                             Op::Lt =>
//                                 self.emit(crate::ir::test::tac::Stmt::Assign(variable_id, crate::ir::test::tac::Op::Lt(lhs, rhs))),
//                             Op::Eq =>
//                                 self.emit(crate::ir::test::tac::Stmt::Assign(variable_id, crate::ir::test::tac::Op::Eq(lhs, rhs))),
//                         };
//                         Val::Var(variable_id)
//                     }
//                 }
//             }
// 
//             fn translate_stmt(&mut self, stmt: Stmt) {
//                 match stmt {
//                     Stmt::Assign(name, expr) => {
//                         let id = self.get_variable(&name);
//                         let val = self.translate_expr(expr);
//                         self.emit(crate::ir::test::tac::Stmt::Assign(id, crate::ir::test::tac::Op::Copy(val)));
//                     }
//                     Stmt::If(cond, then, else_) => {
//                         let cond = self.translate_expr(cond);
//                         match cond {
//                             // optimize out then
//                             Val::Literal(0) => {
//                                 let mut else_block = self.child();
//                                 if let Some(else_) = else_ {
//                                     for stmt in else_ {
//                                         else_block.translate_stmt(stmt);
//                                     }
//                                     let stmts = else_block.finish_block(self);
//                                     for stmt in stmts {
//                                         self.emit(stmt);
//                                     }
//                                 }
//                             }
//                             // optimize out else
//                             Val::Literal(_) => {
//                                 let mut then_block = self.child();
//                                 for stmt in then {
//                                     then_block.translate_stmt(stmt);
//                                 }
//                                 let stmts = then_block.finish_block(self);
//                                 for stmt in stmts {
//                                     self.emit(stmt);
//                                 }
//                             }
//                             Val::Var(cond_id) => {
//                                 let mut then_block = self.child();
//                                 for stmt in then {
//                                     then_block.translate_stmt(stmt);
//                                 }
//                                 let then_stmts = then_block.finish_block(self);
//                                 let mut else_block = self.child();
//                                 if let Some(else_) = else_ {
//                                     for stmt in else_ {
//                                         else_block.translate_stmt(stmt);
//                                     }
//                                 }
//                                 let else_stmts = else_block.finish_block(self);
//                                 if else_stmts.is_empty() {
//                                     self.emit(crate::ir::test::tac::Stmt::If(cond_id, then_stmts, None));
//                                 } else {
//                                     self.emit(crate::ir::test::tac::Stmt::If(cond_id, then_stmts, Some(else_stmts)));
//                                 }
//                             }
//                         };
//                     }
//                     Stmt::While(cond, body) => {
//                         let cond = self.translate_expr(cond);
//                         match cond {
//                             // optimize out body
//                             Val::Literal(0) => {}
//                             // optimize out else
//                             Val::Literal(_) => {
//                                 let mut body_block = self.child();
//                                 for stmt in body {
//                                     body_block.translate_stmt(stmt);
//                                 }
//                                 let stmts = body_block.finish_block(self);
//                                 for stmt in stmts {
//                                     self.emit(stmt);
//                                 }
//                             }
//                             Val::Var(cond_id) => {
//                                 let mut body_block = self.child();
//                                 for stmt in body {
//                                     body_block.translate_stmt(stmt);
//                                 }
//                                 let body_stmts = body_block.finish_block(self);
//                                 self.emit(crate::ir::test::tac::Stmt::While(cond_id, body_stmts));
//                             }
//                         };
//                     }
//                     Stmt::Return(expr) => {
//                         let val = self.translate_expr(expr);
//                         let id = if let Val::Var(id) = val {
//                             id
//                         } else {
//                             let id = self.new_variable();
//                             self.emit(crate::ir::test::tac::Stmt::Assign(id, crate::ir::test::tac::Op::Copy(val)));
//                             id
//                         };
//                         self.emit(crate::ir::test::tac::Stmt::Return(id));
//                     }
//                 }
//             }
//         }
// 
//         #[test]
//         fn blah() {
//             // # Program
//             // x = 1;
//             // y = 2137;
//             // while (x < 10) {
//             //     x = x + 1;
//             //     if (y % 2 == 0) {
//             //        y = y + 1;
//             //     }
//             //     if (y % 5 == 1) {
//             //        return y;
//             //     }
//             //     y = (y % x) + x;
//             // }
//             // return y;
//             // # Converted to SSA
//             // x0 = phi(0, x1);
//             // y0 = phi(2137, y3);
//             // while (x0 < 10) {
//             //     x1 = x0 + 1;
//             //     if (y0 % 2 == 0) {
//             //        y1 = y0 + 1;
//             //     }
//             //     y2 = phi(y1, y0);
//             //     if (y2 % 5 == 1) {
//             //        return y2;
//             //     }
//             //     y3 = y2 % x1 + x1;
//             // }
// 
// 
//             let program = vec![
//                 Stmt::Assign("x".to_string(), Expr::Literal(1)),
//                 Stmt::Assign("y".to_string(), Expr::Literal(2137)),
//                 Stmt::While(
//                     Expr::Binary(Op::Lt, Box::new(Expr::Var("x".to_string())), Box::new(Expr::Literal(10))),
//                     vec![
//                         Stmt::Assign("x".to_string(), Expr::Binary(Op::Add, Box::new(Expr::Var("x".to_string())), Box::new(Expr::Literal(1)))),
//                         Stmt::If(
//                             Expr::Binary(Op::Eq, Box::new(Expr::Binary(Op::Mod, Box::new(Expr::Var("y".to_string())), Box::new(Expr::Literal(2)))), Box::new(Expr::Literal(0))),
//                             vec![
//                                 Stmt::Assign("y".to_string(), Expr::Binary(Op::Add, Box::new(Expr::Var("y".to_string())), Box::new(Expr::Literal(1)))),
//                             ],
//                             None
//                         ),
//                         Stmt::If(
//                             Expr::Binary(Op::Eq, Box::new(Expr::Binary(Op::Mod, Box::new(Expr::Var("y".to_string())), Box::new(Expr::Literal(5)))), Box::new(Expr::Literal(1))),
//                             vec![
//                                 Stmt::Return(Expr::Var("y".to_string())),
//                             ],
//                             None
//                         ),
//                         Stmt::Assign("y".to_string(), Expr::Binary(Op::Add, Box::new(Expr::Binary(Op::Mod, Box::new(Expr::Var("y".to_string())), Box::new(Expr::Var("x".to_string())))), Box::new(Expr::Var("x".to_string())))),
//                     ]
//                 ),
//                 Stmt::Return(Expr::Var("y".to_string())),
//             ];
//             let mut ctx = Context::new();
//             for stmt in program {
//                 ctx.translate_stmt(stmt);
//             }
//             let tac = ctx.finish_block(&mut Context::new());
//             println!("{:#?}", tac);
//         }
//     }
// 
//     mod tac {
//         use crate::ir::test::ast::Val;
//         use crate::ir::VariableId;
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         pub enum Op {
//             Add(crate::ir::test::ast::Val, crate::ir::test::ast::Val),
//             Mod(crate::ir::test::ast::Val, crate::ir::test::ast::Val),
//             Lt(crate::ir::test::ast::Val, crate::ir::test::ast::Val),
//             Eq(crate::ir::test::ast::Val, crate::ir::test::ast::Val),
//             Copy(crate::ir::test::ast::Val),
//         }
// 
//         #[derive(Debug, Clone, PartialEq, Eq)]
//         pub(crate) enum Stmt {
//             Assign(VariableId, Op),
//             If(VariableId, Vec<Stmt>, Option<Vec<Stmt>>),
//             While(VariableId, Vec<Stmt>),
//             Return(VariableId),
//         }
//     }
// }