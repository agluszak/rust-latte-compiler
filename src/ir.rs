use crate::ast;
use crate::ast::Literal;
use crate::ir::BasicBlockContinuation::{ContinueBlock, Stop};
use crate::typechecker::Type;
use crate::typed_ast::{TypedBlock, TypedExpr, TypedExprKind, TypedFnDecl, TypedStmt, VariableId};
use std::collections::BTreeMap;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BlockId(pub(crate) u32);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ValueId(pub(crate) u32);

impl Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
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

impl Value {
    pub(crate) fn rewrite_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Value::Call(_, args) => {
                for arg in args {
                    *arg = f(*arg);
                }
            }
            Value::BinaryOp(_, lhs, rhs) => {
                *lhs = f(*lhs);
                *rhs = f(*rhs);
            }
            Value::UnaryOp(_, operand) => *operand = f(*operand),
            Value::Phi(phi) => {
                for (_, value) in &mut phi.incoming {
                    *value = f(*value);
                }
            }
            Value::Int(_)
            | Value::String(_)
            | Value::Bool(_)
            | Value::Argument(_)
            | Value::Undef => {}
        }
    }
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

impl Terminator {
    pub(crate) fn successors(&self) -> impl Iterator<Item = BlockId> {
        let successors = match *self {
            Terminator::Return(_) | Terminator::ReturnNoValue => [None, None],
            Terminator::Branch(_, then_block, else_block) => [Some(then_block), Some(else_block)],
            Terminator::Jump(target) => [Some(target), None],
        };
        successors.into_iter().flatten()
    }

    pub(crate) fn rewrite_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Terminator::Return(value) => *value = f(*value),
            Terminator::Branch(condition, _, _) => *condition = f(*condition),
            Terminator::ReturnNoValue | Terminator::Jump(_) => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IrBuilder {
    next_value_id: u32,
    next_block_id: u32,
    current_definitions: BTreeMap<VariableId, BTreeMap<BlockId, ValueId>>,
    variable_types: BTreeMap<VariableId, Type>,
    values: BTreeMap<ValueId, BuildingValueData>,
    aliases: BTreeMap<ValueId, ValueId>,
    blocks: BTreeMap<BlockId, BuildingBlock>,
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
    pub entry: BlockId,
    pub values: BTreeMap<ValueId, ValueData>,
    pub blocks: BTreeMap<BlockId, BasicBlock>,
}

impl IrBuilder {
    fn new() -> Self {
        Self {
            next_value_id: 0,
            next_block_id: 0,
            current_definitions: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            values: BTreeMap::new(),
            aliases: BTreeMap::new(),
            blocks: BTreeMap::new(),
            incomplete_phis: BTreeMap::new(),
        }
    }

    fn new_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    fn new_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    fn allocate(&mut self, kind: BuildingValue, ty: Type) -> ValueId {
        let id = self.new_value_id();
        self.values.insert(id, BuildingValueData { ty, kind });
        id
    }

    fn emit(&mut self, block: BlockId, kind: BuildingValue, ty: Type) -> ValueId {
        assert!(self.blocks[&block].terminator.is_none());
        let id = self.allocate(kind, ty);
        self.blocks.get_mut(&block).unwrap().instructions.push(id);
        id
    }

    fn new_phi(&mut self, block: BlockId, ty: Type) -> ValueId {
        let id = self.allocate(BuildingValue::Phi(BuildingPhi::new(block)), ty);
        self.blocks.get_mut(&block).unwrap().phis.push(id);
        id
    }

    fn resolve_alias(&mut self, id: ValueId) -> ValueId {
        let Some(&next) = self.aliases.get(&id) else {
            return id;
        };
        let resolved = self.resolve_alias(next);
        self.aliases.insert(id, resolved);
        resolved
    }

    fn finish_block(&mut self, block_id: BlockId, terminator: Terminator) {
        assert!(self.blocks[&block_id].terminator.is_none());
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
        self.blocks.get_mut(&block_id).unwrap().terminator = Some(terminator);
    }

    fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        let block = self.blocks.get_mut(&block).unwrap();
        assert!(!block.sealed);
        if !block.predecessors.contains(&pred) {
            block.predecessors.push(pred);
        }
    }

    fn new_block(&mut self) -> BlockId {
        let id = self.new_block_id();
        self.blocks.insert(
            id,
            BuildingBlock {
                phis: Vec::new(),
                instructions: Vec::new(),
                terminator: None,
                predecessors: Vec::new(),
                sealed: false,
            },
        );
        id
    }

    fn write_variable(&mut self, variable: VariableId, block: BlockId, value: ValueId) {
        let value = self.resolve_alias(value);
        let value_ty = self.values[&value].ty.clone();
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
            let sealed = self.blocks[&block_id].sealed;
            let predecessors = self.blocks[&block_id].predecessors.clone();
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
        for pred in self.blocks[&phi_block].predecessors.clone() {
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
        match &mut self.values.get_mut(&phi_id)?.kind {
            BuildingValue::Phi(phi) => Some(phi),
            _ => None,
        }
    }

    fn seal_block(&mut self, block_id: BlockId) {
        assert!(!self.blocks[&block_id].sealed);
        self.blocks.get_mut(&block_id).unwrap().sealed = true;
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
            let ty = self.values[&phi_id].ty.clone();
            let undef = self.allocate(BuildingValue::Undef, ty);
            self.blocks
                .get_mut(&phi.block)
                .unwrap()
                .instructions
                .push(undef);
            same = Some(undef);
        }
        // Remember all users except the phi itself
        phi.users.retain(|&user| user != phi_id);
        let replacement = self.resolve_alias(same.unwrap());
        self.aliases.insert(phi_id, replacement);

        // Transfer the users to the replacement phi, so that if the
        // replacement itself becomes trivial later, these users are
        // reconsidered as well.
        if let Some(BuildingValue::Phi(replacement_phi)) =
            self.values.get_mut(&replacement).map(|data| &mut data.kind)
        {
            for user in &phi.users {
                replacement_phi.add_user(*user);
            }
        }

        // Try to recursively remove all phi users, which might have become trivial
        for &user in &phi.users {
            let user = self.resolve_alias(user);
            if matches!(self.values[&user].kind, BuildingValue::Phi(_)) {
                self.try_remove_trivial_phi(user);
            }
        }
        replacement
    }

    fn finish(mut self, ty: Type, entry: BlockId) -> FunctionIr {
        assert!(self.incomplete_phis.is_empty());
        assert!(self.blocks.values().all(|block| block.sealed));
        assert!(self.blocks.values().all(|block| block.terminator.is_some()));

        // Path-compress every alias so surviving operands resolve in one step.
        let aliased: Vec<ValueId> = self.aliases.keys().copied().collect();
        for id in aliased {
            self.resolve_alias(id);
        }

        let aliases = std::mem::take(&mut self.aliases);
        let remap = |id: ValueId| {
            let mut id = id;
            while let Some(&next) = aliases.get(&id) {
                id = next;
            }
            id
        };

        let building_values = std::mem::take(&mut self.values);
        let values = building_values
            .into_iter()
            .filter_map(|(id, data)| {
                if aliases.contains_key(&id) {
                    // The value was replaced by another one; the finalized IR
                    // simply does not contain it anymore.
                    return None;
                }
                Some((
                    id,
                    ValueData {
                        ty: data.ty,
                        kind: finalize_value(data.kind, &remap),
                    },
                ))
            })
            .collect();

        let building_blocks = std::mem::take(&mut self.blocks);
        let blocks = building_blocks
            .into_iter()
            .map(|(id, block)| {
                let operands = |ids: Vec<ValueId>| -> Vec<ValueId> {
                    ids.into_iter()
                        .filter(|id| !aliases.contains_key(id))
                        .map(remap)
                        .collect()
                };
                (
                    id,
                    BasicBlock {
                        phis: operands(block.phis),
                        instructions: operands(block.instructions),
                        terminator: match block.terminator.unwrap() {
                            Terminator::Return(value) => Terminator::Return(remap(value)),
                            Terminator::ReturnNoValue => Terminator::ReturnNoValue,
                            Terminator::Branch(condition, then_block, else_block) => {
                                Terminator::Branch(remap(condition), then_block, else_block)
                            }
                            Terminator::Jump(target) => Terminator::Jump(target),
                        },
                    },
                )
            })
            .collect();

        FunctionIr {
            ty,
            entry,
            values,
            blocks,
        }
    }
}

fn finalize_value(kind: BuildingValue, remap: &impl Fn(ValueId) -> ValueId) -> Value {
    match kind {
        BuildingValue::Int(value) => Value::Int(value),
        BuildingValue::String(value) => Value::String(value),
        BuildingValue::Bool(value) => Value::Bool(value),
        BuildingValue::Argument(index) => Value::Argument(index),
        BuildingValue::Call(function, args) => {
            Value::Call(function, args.into_iter().map(remap).collect())
        }
        BuildingValue::BinaryOp(op, lhs, rhs) => Value::BinaryOp(op, remap(lhs), remap(rhs)),
        BuildingValue::UnaryOp(op, operand) => Value::UnaryOp(op, remap(operand)),
        BuildingValue::Phi(phi) => Value::Phi(Phi {
            incoming: phi
                .incoming
                .into_iter()
                .map(|(block, value)| (block, remap(value)))
                .collect(),
        }),
        BuildingValue::Undef => Value::Undef,
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
        let mut function_ir = ir.finish(ty, entry_block);
        crate::gvn::optimize(&mut function_ir);

        self.functions.insert(function_name, function_ir);
    }

    pub fn dump(&self) -> String {
        let mut result = String::new();
        for (name, function) in &self.functions {
            result.push_str(&format!("Function {}\n", name));
            for (id, block) in &function.blocks {
                result.push_str(&format!("{}:\n", id));
                for phi in &block.phis {
                    result.push_str(&format!("  {:?}: {:?}\n", phi, function.values[phi]));
                }
                for instr in &block.instructions {
                    result.push_str(&format!("  {:?}: {:?}\n", instr, function.values[instr]));
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
                if let BuildingValue::Bool(constant) = &context.values[&cond].kind {
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
                if let BuildingValue::Bool(constant) = &context.values[&cond].kind {
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
        let valid = |id: ValueId| ir.values.contains_key(&id);
        for value in ir.values.values() {
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
        let mut predecessors: BTreeMap<BlockId, BTreeSet<BlockId>> =
            ir.blocks.keys().map(|id| (*id, BTreeSet::new())).collect();
        for (id, block) in &ir.blocks {
            match block.terminator {
                Terminator::Return(_) | Terminator::ReturnNoValue => {}
                Terminator::Branch(_, then_block, else_block) => {
                    predecessors.entry(then_block).or_default().insert(*id);
                    predecessors.entry(else_block).or_default().insert(*id);
                }
                Terminator::Jump(target) => {
                    predecessors.entry(target).or_default().insert(*id);
                }
            }
        }

        for (id, block) in &ir.blocks {
            for &phi in &block.phis {
                assert!(valid(phi));
                assert!(
                    matches!(ir.values[&phi].kind, Value::Phi(_)),
                    "block.phis must only contain phi values"
                );
            }
            for &instr in &block.instructions {
                assert!(valid(instr));
                assert!(
                    !matches!(ir.values[&instr].kind, Value::Phi(_)),
                    "phi {:?} must live in block.phis, not instructions",
                    instr
                );
            }

            // Every phi's incoming edges must match the block's real CFG
            // predecessors exactly: one incoming edge per predecessor.
            for &phi in &block.phis {
                let Value::Phi(phi_data) = &ir.values[&phi].kind else {
                    unreachable!()
                };
                let incoming_blocks: BTreeSet<BlockId> =
                    phi_data.incoming.iter().map(|(block, _)| *block).collect();
                assert_eq!(
                    incoming_blocks, predecessors[id],
                    "phi {:?} incoming edges do not match CFG predecessors of {}",
                    phi, id
                );
                assert_eq!(
                    phi_data.incoming.len(),
                    predecessors[id].len(),
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

    fn assert_values_are_single_definitions(ir: &FunctionIr) {
        let mut seen = BTreeSet::new();
        for block in ir.blocks.values() {
            for &id in block.phis.iter().chain(block.instructions.iter()) {
                assert!(seen.insert(id), "value {:?} defined twice", id);
            }
        }
    }

    #[test]
    fn recursive_trivial_phis_are_removed_and_values_are_canonicalized() {
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

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)), entry);
        assert!(
            ir.values
                .iter()
                .all(|(_, value)| !matches!(value.kind, Value::Phi(_)))
        );
        assert_eq!(ir.values.len(), 2);
        assert_operands_are_valid(&ir);
        assert_values_are_single_definitions(&ir);
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

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)), entry);
        assert!(
            ir.values
                .iter()
                .all(|(_, value)| !matches!(value.kind, Value::Phi(_))),
            "a trivial phi survived finalization"
        );
        assert_operands_are_valid(&ir);
        assert_values_are_single_definitions(&ir);
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

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)), entry);
        let phis: Vec<_> = ir
            .values
            .values()
            .filter_map(|value| match &value.kind {
                Value::Phi(phi) => Some(phi),
                _ => None,
            })
            .collect();
        assert_eq!(phis.len(), 1);
        assert_eq!(phis[0].incoming.len(), 2);
        assert_operands_are_valid(&ir);
        assert_values_are_single_definitions(&ir);

        // Phis are structural: they live in block.phis, never in instructions.
        let header_block = &ir.blocks[&header];
        assert_eq!(header_block.phis.len(), 1);
        assert!(
            header_block
                .instructions
                .iter()
                .all(|&id| !matches!(ir.values[&id].kind, Value::Phi(_)))
        );
    }
}
