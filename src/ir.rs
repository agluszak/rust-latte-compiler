use crate::ast;
use crate::ast::Literal;
use crate::ir::BasicBlockContinuation::{ContinueBlock, Stop};
use crate::typechecker::Type;
use crate::typed_ast::{TypedBlock, TypedExpr, TypedExprKind, TypedFnDecl, TypedStmt, VariableId};
use std::collections::BTreeMap;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BlockId(u32);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

impl BlockId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(index as u32)
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ValueId(u32);

impl Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl ValueId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinaryOpCode {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
    Neq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnaryOpCode {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub incoming: Vec<(BlockId, ValueId)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildingPhi {
    block: BlockId,
    incoming: Vec<(BlockId, ValueId)>,
    users: Vec<ValueId>,
}

impl BuildingPhi {
    fn new(block: BlockId) -> Self {
        Self {
            incoming: Vec::new(),
            block,
            users: Vec::new(),
        }
    }

    fn add_incoming(&mut self, block: BlockId, value: ValueId) {
        self.incoming.push((block, value));
    }

    fn add_user(&mut self, user: ValueId) {
        if !self.users.contains(&user) {
            self.users.push(user);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(i32),
    String(String),
    Bool(bool),
    Call(VariableId, Vec<ValueId>),
    Argument(u32),
    BinaryOp(BinaryOpCode, ValueId, ValueId),
    UnaryOp(UnaryOpCode, ValueId),
    Phi(Phi),
    Undef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BuildingValue {
    Int(i32),
    String(String),
    Bool(bool),
    Call(VariableId, Vec<ValueId>),
    Argument(u32),
    BinaryOp(BinaryOpCode, ValueId, ValueId),
    UnaryOp(UnaryOpCode, ValueId),
    Phi(BuildingPhi),
    Undef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildingValueData {
    ty: Type,
    kind: BuildingValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueData {
    pub ty: Type,
    pub kind: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildingBlock {
    phis: Vec<ValueId>,
    instructions: Vec<ValueId>,
    terminator: Option<Terminator>,
    predecessors: Vec<BlockId>,
    sealed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return(ValueId),
    ReturnNoValue,
    Branch(ValueId, BlockId, BlockId),
    Jump(BlockId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IrBuilder {
    current_definitions: BTreeMap<VariableId, BTreeMap<BlockId, ValueId>>,
    variable_types: BTreeMap<VariableId, Type>,
    values: Vec<BuildingValueData>,
    aliases: Vec<Option<ValueId>>,
    blocks: Vec<BuildingBlock>,
    incomplete_phis: BTreeMap<BlockId, BTreeMap<VariableId, ValueId>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub phis: Vec<ValueId>,
    pub instructions: Vec<ValueId>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionIr {
    pub ty: Type,
    pub values: Vec<ValueData>,
    pub blocks: Vec<BasicBlock>,
}

impl IrBuilder {
    fn new() -> Self {
        Self {
            current_definitions: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            values: Vec::new(),
            aliases: Vec::new(),
            blocks: Vec::new(),
            incomplete_phis: BTreeMap::new(),
        }
    }

    fn allocate(&mut self, kind: BuildingValue, ty: Type) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(BuildingValueData { ty, kind });
        self.aliases.push(None);
        id
    }

    fn emit(&mut self, block: BlockId, kind: BuildingValue, ty: Type) -> ValueId {
        assert!(self.blocks[block.index()].terminator.is_none());
        let id = self.allocate(kind, ty);
        self.blocks[block.index()].instructions.push(id);
        id
    }

    fn new_phi(&mut self, block: BlockId, ty: Type) -> ValueId {
        let id = self.allocate(BuildingValue::Phi(BuildingPhi::new(block)), ty);
        self.blocks[block.index()].phis.push(id);
        id
    }

    fn resolve_alias(&mut self, id: ValueId) -> ValueId {
        let Some(next) = self.aliases[id.index()] else {
            return id;
        };
        let resolved = self.resolve_alias(next);
        self.aliases[id.index()] = Some(resolved);
        resolved
    }

    fn resolve_alias_readonly(&self, mut id: ValueId) -> ValueId {
        while let Some(next) = self.aliases[id.index()] {
            id = next;
        }
        id
    }

    fn finish_block(&mut self, block_id: BlockId, terminator: Terminator) {
        assert!(self.blocks[block_id.index()].terminator.is_none());
        match &terminator {
            Terminator::Return(_) | Terminator::ReturnNoValue => {}
            Terminator::Branch(_, then, else_) => {
                self.add_predecessor(*then, block_id);
                self.add_predecessor(*else_, block_id);
            }
            Terminator::Jump(target) => {
                self.add_predecessor(*target, block_id);
            }
        }
        self.blocks[block_id.index()].terminator = Some(terminator);
    }

    fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        let block = &mut self.blocks[block.index()];
        assert!(!block.sealed);
        if !block.predecessors.contains(&pred) {
            block.predecessors.push(pred);
        }
    }

    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BuildingBlock {
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator: None,
            predecessors: Vec::new(),
            sealed: false,
        });
        id
    }

    fn write_variable(&mut self, variable: VariableId, block: BlockId, value: ValueId) {
        let value = self.resolve_alias(value);
        let value_ty = self.values[value.index()].ty.clone();
        self.variable_types.insert(variable, value_ty);
        self.current_definitions
            .entry(variable)
            .or_default()
            .insert(block, value);
    }

    fn read_variable(&mut self, variable: VariableId, block_id: BlockId) -> ValueId {
        if let Some(value) = self
            .current_definitions
            .get(&variable)
            .and_then(|map| map.get(&block_id))
            .copied()
        {
            self.resolve_alias(value)
        } else {
            let sealed = self.blocks[block_id.index()].sealed;
            let predecessors = self.blocks[block_id.index()].predecessors.clone();
            let ty = self
                .variable_types
                .get(&variable)
                .cloned()
                .unwrap_or_else(|| panic!("Variable {:?} not found", variable));
            let val = if !sealed {
                // Incomplete CFG
                let id = self.new_phi(block_id, ty);
                self.incomplete_phis
                    .entry(block_id)
                    .or_default()
                    .insert(variable, id);
                id
            } else if predecessors.len() == 1 {
                // Optimize the common case of a single predecessor: no phi needed
                self.read_variable(variable, predecessors[0])
            } else {
                // Break potential cycles with operandless phi
                let val = self.new_phi(block_id, ty);
                self.write_variable(variable, block_id, val);

                self.add_phi_operands(variable, val)
            };
            self.write_variable(variable, block_id, val);
            val
        }
    }

    fn add_phi_operands(&mut self, variable: VariableId, phi_id: ValueId) -> ValueId {
        let phi_id = self.resolve_alias(phi_id);
        let phi_block = self.get_phi(phi_id).unwrap().block;
        for pred in self.blocks[phi_block.index()].predecessors.clone() {
            let pred_val = self.read_variable(variable, pred);
            self.add_phi_incoming(phi_id, pred, pred_val);
        }
        self.try_remove_trivial_phi(phi_id)
    }

    /// Adds `(block, value)` as an incoming edge of the building phi `phi`.
    ///
    /// Centralizes the SSA invariant that registering a phi operand must also
    /// update the use relation when the operand is itself a building phi.
    fn add_phi_incoming(&mut self, phi: ValueId, block: BlockId, value: ValueId) {
        let value = self.resolve_alias(value);

        if let Some(operand_phi) = self.get_phi(value) {
            operand_phi.add_user(phi);
        }

        self.get_phi(phi).unwrap().add_incoming(block, value);
    }

    fn get_phi(&mut self, phi_id: ValueId) -> Option<&mut BuildingPhi> {
        match &mut self.values[phi_id.index()].kind {
            BuildingValue::Phi(phi) => Some(phi),
            _ => None,
        }
    }

    fn seal_block(&mut self, block_id: BlockId) {
        assert!(!self.blocks[block_id.index()].sealed);
        self.blocks[block_id.index()].sealed = true;
        if let Some(incomplete_phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi_id) in incomplete_phis {
                self.add_phi_operands(variable, phi_id);
            }
        }
    }

    fn try_remove_trivial_phi(&mut self, phi_id: ValueId) -> ValueId {
        let phi_id = self.resolve_alias(phi_id);
        let mut phi = self.get_phi(phi_id).cloned().unwrap();
        let mut same = None;
        for &(_, operand) in &phi.incoming {
            let op = self.resolve_alias(operand);
            if op == phi_id {
                continue;
            }
            if let Some(same) = same {
                if op == same {
                    // Another edge carrying the same unique value
                    continue;
                } else {
                    // This phi merges at least two different values, so it's not trivial
                    return phi_id;
                }
            } else {
                same = Some(op);
            }
        }
        if same.is_none() {
            // This phi is unreachable or in the entry block
            let ty = self.values[phi_id.index()].ty.clone();
            let undef = self.allocate(BuildingValue::Undef, ty);
            self.blocks[phi.block.index()].instructions.push(undef);
            same = Some(undef);
        }
        // Remember all users except the phi itself
        phi.users.retain(|&user| user != phi_id);
        let replacement = self.resolve_alias(same.unwrap());
        self.aliases[phi_id.index()] = Some(replacement);

        // Transfer the users to the replacement phi, so that if the
        // replacement itself becomes trivial later, these users are
        // reconsidered as well.
        if let Some(BuildingValue::Phi(replacement_phi)) = self
            .values
            .get_mut(replacement.index())
            .map(|data| &mut data.kind)
        {
            for user in &phi.users {
                replacement_phi.add_user(*user);
            }
        }

        // Try to recursively remove all phi users, which might have become trivial
        for &user in &phi.users {
            let user = self.resolve_alias(user);
            if matches!(self.values[user.index()].kind, BuildingValue::Phi(_)) {
                self.try_remove_trivial_phi(user);
            }
        }
        replacement
    }

    fn finish(mut self, ty: Type) -> FunctionIr {
        assert!(self.incomplete_phis.is_empty());
        assert!(self.blocks.iter().all(|block| block.sealed));
        assert!(self.blocks.iter().all(|block| block.terminator.is_some()));

        for id in 0..self.values.len() {
            self.resolve_alias(ValueId(id as u32));
        }

        let mut canonical_ids = vec![None; self.values.len()];
        let mut next_id = 0;
        for (old_id, final_id) in canonical_ids.iter_mut().enumerate() {
            if self.aliases[old_id].is_none() {
                *final_id = Some(ValueId(next_id));
                next_id += 1;
            }
        }
        let remapped_ids: Vec<ValueId> = (0..self.values.len())
            .map(|id| {
                let canonical = self.resolve_alias_readonly(ValueId(id as u32));
                canonical_ids[canonical.index()].expect("canonical value was discarded")
            })
            .collect();
        let remap = |id: ValueId| remapped_ids[id.index()];

        let values = self
            .values
            .into_iter()
            .enumerate()
            .filter_map(|(old_id, data)| {
                canonical_ids[old_id]?;
                let kind = match data.kind {
                    BuildingValue::Int(value) => Value::Int(value),
                    BuildingValue::String(value) => Value::String(value),
                    BuildingValue::Bool(value) => Value::Bool(value),
                    BuildingValue::Argument(index) => Value::Argument(index),
                    BuildingValue::Call(function, args) => {
                        Value::Call(function, args.into_iter().map(remap).collect())
                    }
                    BuildingValue::BinaryOp(op, lhs, rhs) => {
                        Value::BinaryOp(op, remap(lhs), remap(rhs))
                    }
                    BuildingValue::UnaryOp(op, operand) => Value::UnaryOp(op, remap(operand)),
                    BuildingValue::Phi(phi) => Value::Phi(Phi {
                        incoming: phi
                            .incoming
                            .into_iter()
                            .map(|(block, value)| (block, remap(value)))
                            .collect(),
                    }),
                    BuildingValue::Undef => Value::Undef,
                };
                Some(ValueData { ty: data.ty, kind })
            })
            .collect();

        let blocks = self
            .blocks
            .into_iter()
            .map(|block| BasicBlock {
                phis: block
                    .phis
                    .into_iter()
                    .filter(|id| self.aliases[id.index()].is_none())
                    .map(remap)
                    .collect(),
                instructions: block
                    .instructions
                    .into_iter()
                    .filter(|id| self.aliases[id.index()].is_none())
                    .map(remap)
                    .collect(),
                terminator: match block.terminator.unwrap() {
                    Terminator::Return(value) => Terminator::Return(remap(value)),
                    Terminator::ReturnNoValue => Terminator::ReturnNoValue,
                    Terminator::Branch(condition, then_block, else_block) => {
                        Terminator::Branch(remap(condition), then_block, else_block)
                    }
                    Terminator::Jump(target) => Terminator::Jump(target),
                },
            })
            .collect();

        FunctionIr { ty, values, blocks }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicBlockContinuation {
    ContinueBlock(BlockId),
    Stop,
}

pub struct Ir {
    pub functions: BTreeMap<String, FunctionIr>,
}

impl Default for Ir {
    fn default() -> Self {
        Self::new()
    }
}

impl Ir {
    pub fn new() -> Self {
        Ir {
            functions: BTreeMap::new(),
        }
    }

    pub fn translate_function(&mut self, decl: TypedFnDecl) {
        let mut ir = IrBuilder::new();
        let ty = decl.ty();
        let entry_block = ir.new_block();
        ir.seal_block(entry_block);
        for (arg, i) in decl.args.into_iter().zip(0..) {
            let argument = ir.emit(entry_block, BuildingValue::Argument(i), arg.value.ty);
            ir.write_variable(arg.value.var_id, entry_block, argument);
        }

        let continuation = FunctionIr::translate_block(&mut ir, decl.body.value, entry_block);
        if let ContinueBlock(block_id) = continuation {
            ir.finish_block(block_id, Terminator::ReturnNoValue);
        }
        let function_name = decl.name.value.0;
        let function_ir = ir.finish(ty);

        self.functions.insert(function_name, function_ir);
    }

    pub fn dump(&self) -> String {
        let mut result = String::new();
        for (name, function) in &self.functions {
            result.push_str(&format!("Function {}\n", name));
            for (index, block) in function.blocks.iter().enumerate() {
                let id = BlockId(index as u32);
                result.push_str(&format!("{}:\n", id));
                for phi in &block.phis {
                    result.push_str(&format!(
                        "  {:?}: {:?}\n",
                        phi,
                        function.values[phi.index()]
                    ));
                }
                for instr in &block.instructions {
                    result.push_str(&format!(
                        "  {:?}: {:?}\n",
                        instr,
                        function.values[instr.index()]
                    ));
                }
                result.push_str(&format!("  {:?}\n", block.terminator));
            }
        }
        result
    }
}

impl FunctionIr {
    fn translate_expr(
        context: &mut IrBuilder,
        expr: TypedExpr,
        block_id: BlockId,
    ) -> (ValueId, BlockId) {
        let (value, block_id) = match expr.expr {
            TypedExprKind::Variable(_, id) => (context.read_variable(id, block_id), block_id),
            TypedExprKind::Literal(lit) => {
                let val = match lit {
                    Literal::Int(i) => context.emit(block_id, BuildingValue::Int(i), Type::Int),
                    Literal::String(s) => {
                        context.emit(block_id, BuildingValue::String(s), Type::LatteString)
                    }
                    Literal::Bool(b) => context.emit(block_id, BuildingValue::Bool(b), Type::Bool),
                };
                (val, block_id)
            }
            TypedExprKind::Binary { lhs, op, rhs } => {
                // Logical operators are represented exclusively by short-circuit CFG.
                if matches!(op.value, ast::BinaryOp::And | ast::BinaryOp::Or) {
                    let (lhs, lhs_block) = Self::translate_expr(context, lhs.value, block_id);

                    let rhs_block = context.new_block();
                    let join_block = context.new_block();

                    // `a && b` evaluates b only when a is true, so the direct
                    // edge to the join carries `false` (a itself); `a || b`
                    // evaluates b only when a is false, so its direct edge
                    // carries `true` (again a itself). Either way the phi
                    // merges the actual operand values, no synthetic blocks
                    // or booleans needed.
                    let (then_block, else_block) = match op.value {
                        ast::BinaryOp::And => (rhs_block, join_block),
                        ast::BinaryOp::Or => (join_block, rhs_block),
                        _ => unreachable!(),
                    };

                    context
                        .finish_block(lhs_block, Terminator::Branch(lhs, then_block, else_block));
                    context.seal_block(rhs_block);
                    let (rhs, rhs_end) = Self::translate_expr(context, rhs.value, rhs_block);
                    context.finish_block(rhs_end, Terminator::Jump(join_block));
                    context.seal_block(join_block);

                    let phi = context.new_phi(join_block, Type::Bool);
                    context.add_phi_incoming(phi, lhs_block, lhs);
                    context.add_phi_incoming(phi, rhs_end, rhs);
                    let phi = context.try_remove_trivial_phi(phi);
                    return (phi, join_block);
                }

                let op = match op.value {
                    ast::BinaryOp::Add => BinaryOpCode::Add,
                    ast::BinaryOp::Sub => BinaryOpCode::Sub,
                    ast::BinaryOp::Mul => BinaryOpCode::Mul,
                    ast::BinaryOp::Div => BinaryOpCode::Div,
                    ast::BinaryOp::Mod => BinaryOpCode::Mod,
                    ast::BinaryOp::Gt => BinaryOpCode::Gt,
                    ast::BinaryOp::Lt => BinaryOpCode::Lt,
                    ast::BinaryOp::Gte => BinaryOpCode::Gte,
                    ast::BinaryOp::Lte => BinaryOpCode::Lte,
                    ast::BinaryOp::Eq => BinaryOpCode::Eq,
                    ast::BinaryOp::Neq => BinaryOpCode::Neq,
                    ast::BinaryOp::And | ast::BinaryOp::Or => unreachable!(),
                };

                let (lhs, block_id) = Self::translate_expr(context, lhs.value, block_id);
                let (rhs, block_id) = Self::translate_expr(context, rhs.value, block_id);

                let val = context.emit(block_id, BuildingValue::BinaryOp(op, lhs, rhs), expr.ty);
                (val, block_id)
            }
            TypedExprKind::Unary { op, expr: target } => {
                let (val, block_id) = Self::translate_expr(context, target.value, block_id);
                let op = match op.value {
                    ast::UnaryOp::Neg => UnaryOpCode::Neg,
                    ast::UnaryOp::Not => UnaryOpCode::Not,
                };
                let val = context.emit(block_id, BuildingValue::UnaryOp(op, val), expr.ty);
                (val, block_id)
            }
            TypedExprKind::Application { target, args } => {
                let TypedExprKind::Variable(_, id) = target.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };

                let mut arg_values = Vec::new();
                let mut current_block_id = block_id;
                for arg in args {
                    let (arg, block_id) =
                        Self::translate_expr(context, arg.value, current_block_id);
                    current_block_id = block_id;
                    arg_values.push(arg);
                }

                let val = context.emit(
                    current_block_id,
                    BuildingValue::Call(id, arg_values),
                    expr.ty,
                );
                (val, current_block_id)
            }
        };
        (value, block_id)
    }

    fn translate_block(
        context: &mut IrBuilder,
        block: TypedBlock,
        block_id: BlockId,
    ) -> BasicBlockContinuation {
        let mut block_id = block_id;
        let mut final_continuation = ContinueBlock(block_id);
        for stmt in block.0 {
            let continuation = Self::translate_stmt(context, stmt.value, block_id);
            match continuation {
                ContinueBlock(new_block_id) => {
                    block_id = new_block_id;
                    final_continuation = ContinueBlock(new_block_id);
                }
                Stop => return Stop,
            }
        }
        final_continuation
    }

    fn default_value(ty: &Type) -> BuildingValue {
        match ty {
            Type::Int => BuildingValue::Int(0),
            Type::Bool => BuildingValue::Bool(false),
            Type::LatteString => BuildingValue::String(String::new()),
            Type::Function(_, _) => panic!("Function cannot have a default value"),
            Type::Void => panic!("Void cannot have a default value"),
        }
    }

    fn translate_stmt(
        context: &mut IrBuilder,
        stmt: TypedStmt,
        block_id: BlockId,
    ) -> BasicBlockContinuation {
        match stmt {
            TypedStmt::Empty => ContinueBlock(block_id),
            TypedStmt::Block(block) => Self::translate_block(context, block.value, block_id),
            TypedStmt::Decl(decl) => {
                let mut block_id = block_id;
                for item in decl.value.items {
                    if let Some(expr) = item.value.init {
                        let (expr, continuation_block) =
                            Self::translate_expr(context, expr.value, block_id);
                        block_id = continuation_block;
                        context.write_variable(item.value.var_id, block_id, expr);
                    } else {
                        let default = Self::default_value(&item.value.ty);
                        let default = context.emit(block_id, default, item.value.ty);
                        context.write_variable(item.value.var_id, block_id, default);
                    }
                }
                ContinueBlock(block_id)
            }
            TypedStmt::Assignment {
                target: _,
                target_id,
                expr,
            } => {
                let (expr, block_id) = Self::translate_expr(context, expr.value, block_id);
                context.write_variable(target_id, block_id, expr);
                ContinueBlock(block_id)
            }
            TypedStmt::Return(expr) => {
                let expr = expr.map(|expr| Self::translate_expr(context, expr.value, block_id));
                if let Some((expr, block_id)) = expr {
                    context.finish_block(block_id, Terminator::Return(expr));
                } else {
                    context.finish_block(block_id, Terminator::ReturnNoValue);
                }
                Stop
            }
            TypedStmt::If {
                cond,
                then,
                otherwise,
            } => {
                let (cond, block_id) = Self::translate_expr(context, cond.value, block_id);
                let cond = context.resolve_alias(cond);
                if let BuildingValue::Bool(constant) = &context.values[cond.index()].kind {
                    return if *constant {
                        Self::translate_stmt(context, then.value, block_id)
                    } else if let Some(otherwise) = otherwise {
                        Self::translate_stmt(context, otherwise.value, block_id)
                    } else {
                        ContinueBlock(block_id)
                    };
                }

                let then_block = context.new_block();
                let then_continuation = Self::translate_stmt(context, then.value, then_block);
                if let Some(otherwise) = otherwise {
                    let else_block = context.new_block();
                    let else_continuation =
                        Self::translate_stmt(context, otherwise.value, else_block);
                    context
                        .finish_block(block_id, Terminator::Branch(cond, then_block, else_block));
                    context.seal_block(else_block);
                    context.seal_block(then_block);

                    if let (Stop, Stop) = (then_continuation, else_continuation) {
                        return Stop;
                    }

                    let after_block = context.new_block();

                    if let ContinueBlock(after_then_block) = then_continuation {
                        context.finish_block(after_then_block, Terminator::Jump(after_block));
                    }

                    if let ContinueBlock(after_else_block) = else_continuation {
                        context.finish_block(after_else_block, Terminator::Jump(after_block));
                    }
                    context.seal_block(after_block);

                    ContinueBlock(after_block)
                } else {
                    let after_block = context.new_block();

                    context
                        .finish_block(block_id, Terminator::Branch(cond, then_block, after_block));
                    context.seal_block(then_block);

                    if let ContinueBlock(after_then_block) = then_continuation {
                        context.finish_block(after_then_block, Terminator::Jump(after_block));
                    }

                    context.seal_block(after_block);

                    ContinueBlock(after_block)
                }
            }
            TypedStmt::While { cond, body } => {
                let cond_block = context.new_block();
                context.finish_block(block_id, Terminator::Jump(cond_block));

                let (cond, cond_block) = Self::translate_expr(context, cond.value, cond_block);

                let cond = context.resolve_alias(cond);
                if let BuildingValue::Bool(constant) = &context.values[cond.index()].kind {
                    return if !*constant {
                        let after_block = context.new_block();
                        context.finish_block(cond_block, Terminator::Jump(after_block));
                        context.seal_block(cond_block);
                        context.seal_block(after_block);
                        ContinueBlock(after_block)
                    } else {
                        let body_block = context.new_block();
                        context.finish_block(cond_block, Terminator::Jump(body_block));
                        context.seal_block(body_block);
                        let body_continuation =
                            Self::translate_stmt(context, body.value, body_block);
                        if let ContinueBlock(after_body_block) = body_continuation {
                            context.finish_block(after_body_block, Terminator::Jump(cond_block));
                        }
                        context.seal_block(cond_block);
                        Stop
                    };
                }

                let after_block = context.new_block();
                let body_block = context.new_block();

                context.finish_block(
                    cond_block,
                    Terminator::Branch(cond, body_block, after_block),
                );
                context.seal_block(body_block);
                context.seal_block(after_block);

                let body_continuation = Self::translate_stmt(context, body.value, body_block);
                if let ContinueBlock(after_body_block) = body_continuation {
                    context.finish_block(after_body_block, Terminator::Jump(cond_block));
                }

                context.seal_block(cond_block);
                ContinueBlock(after_block)
            }
            TypedStmt::Expr(expr) => {
                let (_, block_id) = Self::translate_expr(context, expr.value, block_id);
                ContinueBlock(block_id)
            }
            TypedStmt::Incr(expr) => {
                let TypedExprKind::Variable(_, var_id) = expr.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };
                let (expr, block_id) = Self::translate_expr(context, expr.value, block_id);
                let one = context.emit(block_id, BuildingValue::Int(1), Type::Int);
                let op = context.emit(
                    block_id,
                    BuildingValue::BinaryOp(BinaryOpCode::Add, expr, one),
                    Type::Int,
                );
                context.write_variable(var_id, block_id, op);
                ContinueBlock(block_id)
            }
            TypedStmt::Decr(expr) => {
                let TypedExprKind::Variable(_, var_id) = expr.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };
                let (expr, block_id) = Self::translate_expr(context, expr.value, block_id);
                let one = context.emit(block_id, BuildingValue::Int(1), Type::Int);
                let op = context.emit(
                    block_id,
                    BuildingValue::BinaryOp(BinaryOpCode::Sub, expr, one),
                    Type::Int,
                );
                context.write_variable(var_id, block_id, op);
                ContinueBlock(block_id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn assert_operands_are_valid(ir: &FunctionIr) {
        let valid = |id: ValueId| id.index() < ir.values.len();
        for value in &ir.values {
            match &value.kind {
                Value::Call(_, args) => assert!(args.iter().copied().all(valid)),
                Value::BinaryOp(_, lhs, rhs) => assert!(valid(*lhs) && valid(*rhs)),
                Value::UnaryOp(_, operand) => assert!(valid(*operand)),
                Value::Phi(phi) => {
                    assert!(phi.incoming.iter().all(|(_, value)| valid(*value)))
                }
                Value::Int(_)
                | Value::String(_)
                | Value::Bool(_)
                | Value::Argument(_)
                | Value::Undef => {}
            }
        }

        // Predecessor sets, derived from terminators only.
        let mut predecessors: Vec<BTreeSet<BlockId>> = vec![BTreeSet::new(); ir.blocks.len()];
        for (index, block) in ir.blocks.iter().enumerate() {
            let from = BlockId(index as u32);
            match block.terminator {
                Terminator::Return(_) | Terminator::ReturnNoValue => {}
                Terminator::Branch(_, then_block, else_block) => {
                    predecessors[then_block.index()].insert(from);
                    predecessors[else_block.index()].insert(from);
                }
                Terminator::Jump(target) => {
                    predecessors[target.index()].insert(from);
                }
            }
        }

        for (index, block) in ir.blocks.iter().enumerate() {
            for &id in &block.phis {
                assert!(valid(id));
                assert!(
                    matches!(ir.values[id.index()].kind, Value::Phi(_)),
                    "block.phis must only contain phi values"
                );
            }
            for &id in &block.instructions {
                assert!(valid(id));
                assert!(
                    !matches!(ir.values[id.index()].kind, Value::Phi(_)),
                    "phi {:?} must live in block.phis, not instructions",
                    id
                );
            }

            // Every phi's incoming edges must match the block's real CFG
            // predecessors exactly: one incoming edge per predecessor.
            for &phi in &block.phis {
                let Value::Phi(phi_data) = &ir.values[phi.index()].kind else {
                    unreachable!()
                };
                let incoming_blocks: BTreeSet<BlockId> =
                    phi_data.incoming.iter().map(|(block, _)| *block).collect();
                assert_eq!(
                    incoming_blocks,
                    predecessors[index],
                    "phi {:?} incoming edges do not match CFG predecessors of {}",
                    phi,
                    BlockId(index as u32)
                );
                assert_eq!(
                    phi_data.incoming.len(),
                    predecessors[index].len(),
                    "phi {:?} has duplicate incoming edges for one predecessor",
                    phi
                );
            }

            match block.terminator {
                Terminator::Return(value) | Terminator::Branch(value, _, _) => {
                    assert!(valid(value))
                }
                Terminator::ReturnNoValue | Terminator::Jump(_) => {}
            }
        }
    }

    #[test]
    fn recursive_trivial_phis_are_removed_and_values_are_compacted() {
        let mut builder = IrBuilder::new();
        let entry = builder.new_block();
        let left = builder.new_block();
        let right = builder.new_block();
        let join = builder.new_block();
        builder.seal_block(entry);

        let condition = builder.emit(entry, BuildingValue::Bool(true), Type::Bool);
        let value = builder.emit(entry, BuildingValue::Int(7), Type::Int);
        builder.finish_block(entry, Terminator::Branch(condition, left, right));
        builder.seal_block(left);
        builder.seal_block(right);
        builder.finish_block(left, Terminator::Jump(join));
        builder.finish_block(right, Terminator::Jump(join));
        builder.seal_block(join);

        let first = builder.new_phi(join, Type::Int);
        builder.get_phi(first).unwrap().incoming = [(left, value), (right, first)].into();
        let second = builder.new_phi(join, Type::Int);
        builder.get_phi(second).unwrap().incoming = [(left, first), (right, second)].into();
        builder.get_phi(first).unwrap().users.push(second);

        assert_eq!(builder.try_remove_trivial_phi(first), value);
        assert_eq!(builder.resolve_alias(second), value);
        builder.finish_block(join, Terminator::Return(second));

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)));
        assert!(
            ir.values
                .iter()
                .all(|value| !matches!(value.kind, Value::Phi(_)))
        );
        assert_eq!(ir.values.len(), 2);
        assert_operands_are_valid(&ir);
        assert_values_are_dense(&ir);
    }

    fn assert_values_are_dense(ir: &FunctionIr) {
        let mut seen = vec![false; ir.values.len()];
        for block in &ir.blocks {
            for &id in block.phis.iter().chain(block.instructions.iter()) {
                assert!(!seen[id.index()], "value {:?} defined twice", id);
                seen[id.index()] = true;
            }
        }
        assert!(
            seen.iter().all(|&seen| seen),
            "value ids are not dense: {:?}",
            seen
        );
    }

    #[test]
    fn users_are_transferred_when_a_trivial_phi_is_replaced_by_another_phi() {
        let mut builder = IrBuilder::new();
        let entry = builder.new_block();
        let left = builder.new_block();
        let right = builder.new_block();
        let join = builder.new_block();
        builder.seal_block(entry);

        let condition = builder.emit(entry, BuildingValue::Bool(true), Type::Bool);
        let x = builder.emit(entry, BuildingValue::Int(7), Type::Int);
        builder.finish_block(entry, Terminator::Branch(condition, left, right));
        builder.seal_block(left);
        builder.seal_block(right);
        builder.finish_block(left, Terminator::Jump(join));
        builder.finish_block(right, Terminator::Jump(join));
        builder.seal_block(join);

        // B merges the same value on both edges, but is not removed yet.
        let b = builder.new_phi(join, Type::Int);
        builder.get_phi(b).unwrap().incoming = [(left, x), (right, x)].into();
        // A is trivially replaced by B.
        let a = builder.new_phi(join, Type::Int);
        builder.get_phi(a).unwrap().incoming = [(left, b), (right, b)].into();
        // C uses A and X, so it is not trivial while A is still alive.
        let c = builder.new_phi(join, Type::Int);
        builder.get_phi(c).unwrap().incoming = [(left, a), (right, x)].into();
        builder.get_phi(a).unwrap().add_user(c);

        // A -> B. C must be transferred to B's users.
        assert_eq!(builder.try_remove_trivial_phi(a), b);

        // B -> X later. C has effectively become phi(X, X) and must go too.
        assert_eq!(builder.try_remove_trivial_phi(b), x);
        assert_eq!(builder.resolve_alias(c), x);

        builder.finish_block(join, Terminator::Return(c));

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)));
        assert!(
            ir.values
                .iter()
                .all(|value| !matches!(value.kind, Value::Phi(_))),
            "a trivial phi survived finalization"
        );
        assert_operands_are_valid(&ir);
        assert_values_are_dense(&ir);
    }

    #[test]
    fn loop_header_incomplete_phi_becomes_canonical_ssa() {
        let mut builder = IrBuilder::new();
        let entry = builder.new_block();
        let header = builder.new_block();
        let body = builder.new_block();
        let exit = builder.new_block();
        builder.seal_block(entry);
        let variable = VariableId::new(0);

        let initial = builder.emit(entry, BuildingValue::Int(0), Type::Int);
        builder.write_variable(variable, entry, initial);
        builder.finish_block(entry, Terminator::Jump(header));

        let header_value = builder.read_variable(variable, header);
        assert!(builder.incomplete_phis[&header].contains_key(&variable));
        let condition = builder.emit(header, BuildingValue::Bool(true), Type::Bool);
        builder.finish_block(header, Terminator::Branch(condition, body, exit));
        builder.seal_block(body);
        builder.seal_block(exit);

        let body_value = builder.read_variable(variable, body);
        let one = builder.emit(body, BuildingValue::Int(1), Type::Int);
        let next = builder.emit(
            body,
            BuildingValue::BinaryOp(BinaryOpCode::Add, body_value, one),
            Type::Int,
        );
        builder.write_variable(variable, body, next);
        builder.finish_block(body, Terminator::Jump(header));
        builder.seal_block(header);
        builder.finish_block(exit, Terminator::Return(header_value));

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)));
        let phis: Vec<_> = ir
            .values
            .iter()
            .filter_map(|value| match &value.kind {
                Value::Phi(phi) => Some(phi),
                _ => None,
            })
            .collect();
        assert_eq!(phis.len(), 1);
        assert_eq!(phis[0].incoming.len(), 2);
        assert_operands_are_valid(&ir);
        assert_values_are_dense(&ir);

        // Phis are structural: they live in block.phis, never in instructions.
        let header_block = &ir.blocks[header.index()];
        assert_eq!(header_block.phis.len(), 1);
        assert!(
            header_block
                .instructions
                .iter()
                .all(|&id| !matches!(ir.values[id.index()].kind, Value::Phi(_)))
        );
    }
}
