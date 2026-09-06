use crate::ir::{BinaryOpCode, BlockId, FunctionIr, Terminator, UnaryOpCode, Value, ValueId};
use crate::symbols::{
    RUNTIME_NEW_STRING, RUNTIME_STRING_CONCAT, RUNTIME_STRING_EQUAL, language_builtins,
    mangle_user, resolve_callee,
};
use crate::typechecker::ReadyEnvironment;
use crate::types::Type;
use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicType, BasicTypeEnum, FunctionType, StructType};
use inkwell::values::{BasicValue, BasicValueEnum, GlobalValue, PhiValue};
use std::collections::BTreeMap;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    string_type: StructType<'ctx>,
    env: ReadyEnvironment,
    string_globals: BTreeMap<String, GlobalValue<'ctx>>,
    next_string_id: u32,
}
impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context, name: &str, env: ReadyEnvironment) -> Self {
        let module = context.create_module(name);
        let builder = context.create_builder();

        let string_type = context.opaque_struct_type("string");
        string_type.set_body(
            &[
                context.i8_type().ptr_type(AddressSpace::default()).into(),
                context.i32_type().into(),
            ],
            false,
        );

        let codegen = CodeGen {
            context,
            module,
            builder,
            string_type,
            env,
            string_globals: BTreeMap::new(),
            next_string_id: 0,
        };
        codegen.declare_builtins();

        codegen
    }

    pub fn declare_builtins(&self) {
        let i8_type = self.context.i8_type();
        let i32_type = self.context.i32_type();
        let void = self.context.void_type();
        let string_ptr = self.string_type.ptr_type(AddressSpace::default());

        for (name, args, ret) in language_builtins() {
            let params: Vec<_> = args.iter().map(|arg| self.llvm_basic_type(arg).into()).collect();
            let fn_type = match ret {
                Type::Void => void.fn_type(&params, false),
                _ => self
                    .llvm_basic_type(&ret)
                    .fn_type(&params, false),
            };
            self.module
                .add_function(name, fn_type, Some(Linkage::External));
        }

        self.module.add_function(
            RUNTIME_NEW_STRING,
            string_ptr.fn_type(
                &[
                    i8_type.ptr_type(AddressSpace::default()).into(),
                    i32_type.into(),
                ],
                false,
            ),
            Some(Linkage::External),
        );

        self.module.add_function(
            RUNTIME_STRING_CONCAT,
            string_ptr.fn_type(&[string_ptr.into(), string_ptr.into()], false),
            Some(Linkage::External),
        );

        self.module.add_function(
            RUNTIME_STRING_EQUAL,
            i32_type.fn_type(&[string_ptr.into(), string_ptr.into()], false),
            Some(Linkage::External),
        );
    }

    fn llvm_basic_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Int => self.context.i32_type().into(),
            Type::Bool => self.context.bool_type().into(),
            Type::Void => panic!("void type is not a basic llvm type"),
            Type::Function(_, _) => panic!("function type is not a basic llvm type"),
            Type::LatteString => self.string_type.ptr_type(AddressSpace::default()).into(),
        }
    }

    fn llvm_function_type(&self, ty: &Type) -> FunctionType<'ctx> {
        match ty {
            Type::Function(args, ret) => {
                let args = args
                    .iter()
                    .map(|arg| self.llvm_basic_type(arg).into())
                    .collect::<Vec<_>>();
                match ret.as_ref() {
                    Type::Void => self.context.void_type().fn_type(&args, false),
                    ret => self.llvm_basic_type(ret).fn_type(&args, false),
                }
            }
            _ => panic!("not a function type"),
        }
    }

    fn llvm_undef(&self, ty: &Type) -> BasicValueEnum<'ctx> {
        match ty {
            Type::Bool => self.context.bool_type().get_undef().into(),
            Type::Int => self.context.i32_type().get_undef().into(),
            Type::LatteString => self
                .string_type
                .ptr_type(AddressSpace::default())
                .get_undef()
                .into(),
            Type::Function(_, _) | Type::Void => panic!("invalid undef value type"),
        }
    }

    fn int_comparison_predicate(op: BinaryOpCode) -> Option<IntPredicate> {
        match op {
            BinaryOpCode::Gt => Some(IntPredicate::SGT),
            BinaryOpCode::Lt => Some(IntPredicate::SLT),
            BinaryOpCode::Gte => Some(IntPredicate::SGE),
            BinaryOpCode::Lte => Some(IntPredicate::SLE),
            BinaryOpCode::Eq => Some(IntPredicate::EQ),
            BinaryOpCode::Neq => Some(IntPredicate::NE),
            _ => None,
        }
    }

    fn string_bytes(&mut self, s: &str) -> GlobalValue<'ctx> {
        if let Some(global) = self.string_globals.get(s) {
            return *global;
        }
        let bytes = s.as_bytes();
        // Empty literals still need a valid non-null base pointer for the
        // `newString` call, even though zero bytes are copied.
        let (array_type, initializer) = if bytes.is_empty() {
            (
                self.context.i8_type().array_type(1),
                self.context.const_string(b"\x00", false),
            )
        } else {
            (
                self.context.i8_type().array_type(bytes.len() as u32),
                self.context.const_string(bytes, false),
            )
        };
        let id = self.next_string_id;
        self.next_string_id += 1;
        let global = self.module.add_global(
            array_type,
            Some(AddressSpace::default()),
            &format!("latte.str.{id}"),
        );
        global.set_initializer(&initializer);
        global.set_constant(true);
        global.set_linkage(Linkage::Private);
        self.string_globals.insert(s.to_string(), global);
        global
    }

    fn emit_string_equality(
        &self,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
        sense: IntPredicate,
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let string_equal = self
            .builder
            .build_call(
                self.module.get_function(RUNTIME_STRING_EQUAL).unwrap(),
                &[lhs.into(), rhs.into()],
                "strings_equal",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        self.builder
            .build_int_compare(
                sense,
                string_equal,
                string_equal.get_type().const_zero(),
                name,
            )
            .unwrap()
            .into()
    }

    #[allow(clippy::too_many_lines)]
    fn emit_value(
        &mut self,
        id: ValueId,
        ir: &FunctionIr,
        values: &BTreeMap<ValueId, BasicValueEnum<'ctx>>,
        function: inkwell::values::FunctionValue<'ctx>,
    ) -> Option<BasicValueEnum<'ctx>> {
        let data = &ir.values[&id];
        match &data.kind {
            Value::Int(i) => Some(self.context.i32_type().const_int(*i as u64, true).into()),
            Value::String(s) => {
                let len = self.context.i32_type().const_int(s.len() as u64, false);
                let global = self.string_bytes(s);
                let str_ptr = self
                    .builder
                    .build_bit_cast(
                        global.as_pointer_value(),
                        self.context.i8_type().ptr_type(AddressSpace::default()),
                        "str_ptr",
                    )
                    .unwrap();
                let new_string_fn = self.module.get_function(RUNTIME_NEW_STRING).unwrap();
                let string_ptr = self
                    .builder
                    .build_call(new_string_fn, &[str_ptr.into(), len.into()], "new_string")
                    .unwrap();
                Some(string_ptr.try_as_basic_value().unwrap_basic())
            }
            Value::Bool(b) => Some(
                self.context
                    .bool_type()
                    .const_int(*b as u64, false)
                    .into(),
            ),
            Value::Call(var_id, args) => {
                let source_name = &self.env.names[var_id];
                let callee = resolve_callee(source_name);
                let function = self.module.get_function(&callee).unwrap();
                let args = args
                    .iter()
                    .map(|arg| values[arg].into())
                    .collect::<Vec<_>>();
                let call = self
                    .builder
                    .build_call(function, args.as_slice(), &callee)
                    .unwrap();
                call.try_as_basic_value().basic()
            }
            Value::Argument(i) => Some(function.get_nth_param(*i).unwrap()),
            Value::BinaryOp(op, lhs, rhs) => Some(self.emit_binary(*op, id, *lhs, *rhs, ir, values)),
            Value::UnaryOp(op, operand) => {
                let operand = values[operand].into_int_value();
                Some(match op {
                    UnaryOpCode::Neg => self
                        .builder
                        .build_int_neg(operand, &id.to_string())
                        .unwrap()
                        .into(),
                    UnaryOpCode::Not => self
                        .builder
                        .build_not(operand, &id.to_string())
                        .unwrap()
                        .into(),
                })
            }
            Value::Phi(_) => unreachable!("phi values live in block.phis"),
            Value::Undef => Some(self.llvm_undef(&data.ty)),
        }
    }

    fn emit_binary(
        &self,
        op: BinaryOpCode,
        id: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        ir: &FunctionIr,
        values: &BTreeMap<ValueId, BasicValueEnum<'ctx>>,
    ) -> BasicValueEnum<'ctx> {
        let name = id.to_string();
        match op {
            BinaryOpCode::Add if ir.values[&lhs].ty == Type::LatteString => {
                let lhs = values[&lhs].into_pointer_value();
                let rhs = values[&rhs].into_pointer_value();
                let concat = self.module.get_function(RUNTIME_STRING_CONCAT).unwrap();
                self.builder
                    .build_call(concat, &[lhs.into(), rhs.into()], "new_string")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
            }
            BinaryOpCode::Add => {
                let lhs = values[&lhs].into_int_value();
                let rhs = values[&rhs].into_int_value();
                self.builder
                    .build_int_add(lhs, rhs, &name)
                    .unwrap()
                    .into()
            }
            BinaryOpCode::Sub => {
                let lhs = values[&lhs].into_int_value();
                let rhs = values[&rhs].into_int_value();
                self.builder
                    .build_int_sub(lhs, rhs, &name)
                    .unwrap()
                    .into()
            }
            BinaryOpCode::Mul => {
                let lhs = values[&lhs].into_int_value();
                let rhs = values[&rhs].into_int_value();
                self.builder
                    .build_int_mul(lhs, rhs, &name)
                    .unwrap()
                    .into()
            }
            BinaryOpCode::Div => {
                let lhs = values[&lhs].into_int_value();
                let rhs = values[&rhs].into_int_value();
                self.builder
                    .build_int_signed_div(lhs, rhs, &name)
                    .unwrap()
                    .into()
            }
            BinaryOpCode::Mod => {
                let lhs = values[&lhs].into_int_value();
                let rhs = values[&rhs].into_int_value();
                self.builder
                    .build_int_signed_rem(lhs, rhs, &name)
                    .unwrap()
                    .into()
            }
            BinaryOpCode::Gt | BinaryOpCode::Lt | BinaryOpCode::Gte | BinaryOpCode::Lte => {
                let lhs = values[&lhs].into_int_value();
                let rhs = values[&rhs].into_int_value();
                let predicate = Self::int_comparison_predicate(op).unwrap();
                self.builder
                    .build_int_compare(predicate, lhs, rhs, &name)
                    .unwrap()
                    .into()
            }
            BinaryOpCode::Eq | BinaryOpCode::Neq => {
                if ir.values[&lhs].ty == Type::LatteString {
                    let sense = match op {
                        BinaryOpCode::Eq => IntPredicate::NE,
                        BinaryOpCode::Neq => IntPredicate::EQ,
                        _ => unreachable!(),
                    };
                    self.emit_string_equality(values[&lhs], values[&rhs], sense, &name)
                } else {
                    let lhs = values[&lhs].into_int_value();
                    let rhs = values[&rhs].into_int_value();
                    let predicate = Self::int_comparison_predicate(op).unwrap();
                    self.builder
                        .build_int_compare(predicate, lhs, rhs, &name)
                        .unwrap()
                        .into()
                }
            }
        }
    }

    pub fn declare(&self, name: &str, ir: &FunctionIr) {
        let fn_type = self.llvm_function_type(&ir.ty);
        self.module
            .add_function(&mangle_user(name), fn_type, None);
    }

    pub fn generate(&mut self, name: &str, ir: &FunctionIr) {
        let mangled = mangle_user(name);
        let function = self.module.get_function(&mangled).unwrap();
        let block_order = crate::cfg::reachable_reverse_postorder(ir);
        let basic_blocks: BTreeMap<BlockId, BasicBlock> = block_order
            .iter()
            .map(|id| {
                (
                    *id,
                    self.context.append_basic_block(function, &id.to_string()),
                )
            })
            .collect();
        let mut values: BTreeMap<ValueId, BasicValueEnum> = BTreeMap::new();
        let mut phis: BTreeMap<ValueId, PhiValue> = BTreeMap::new();
        for id in &block_order {
            let block = &ir.blocks[id];
            let this_block = basic_blocks[&id];
            self.builder.position_at_end(this_block);
            // Phis come first, so their values are available to all instructions.
            for &value in &block.phis {
                let value_data = &ir.values[&value];
                debug_assert!(matches!(value_data.kind, Value::Phi(_)));
                // Incoming values will be set later
                let llvm_phi = self
                    .builder
                    .build_phi(self.llvm_basic_type(&value_data.ty), &value.to_string())
                    .unwrap();
                phis.insert(value, llvm_phi);

                values.insert(value, llvm_phi.as_basic_value());
            }

            // Then the rest; void calls produce no value.
            for &value in &block.instructions {
                if let Some(result) = self.emit_value(value, ir, &values, function) {
                    values.insert(value, result);
                }
            }
            match block.terminator {
                Terminator::Return(val) => {
                    let val = values[&val];
                    self.builder.build_return(Some(&val)).unwrap();
                }
                Terminator::ReturnNoValue => {
                    self.builder.build_return(None).unwrap();
                }
                Terminator::Branch(val, then, else_) => {
                    let val = values[&val].into_int_value();
                    let then = basic_blocks[&then];
                    let else_ = basic_blocks[&else_];
                    self.builder
                        .build_conditional_branch(val, then, else_)
                        .unwrap();
                }
                Terminator::Jump(target) => {
                    let target = basic_blocks[&target];
                    self.builder.build_unconditional_branch(target).unwrap();
                }
            }
        }

        // The incoming-edge pass reads the authoritative `Value::Phi` data from
        // the IR and pairs it with the LLVM phis created earlier.
        for block in ir.blocks.values() {
            for &value in &block.phis {
                let llvm_phi = &phis[&value];
                let Value::Phi(phi) = &ir.values[&value].kind else {
                    unreachable!("block.phis must only contain phi values");
                };

                let mut incoming: Vec<(BasicValueEnum, BasicBlock)> = Vec::new();
                for (block, operand) in &phi.incoming {
                    let operand = values[operand];
                    incoming.push((operand, basic_blocks[block]));
                }
                let incoming: Vec<(&dyn BasicValue, BasicBlock)> = incoming
                    .iter()
                    .map(|(value, block)| (value as &dyn BasicValue, *block))
                    .collect();

                llvm_phi.add_incoming(incoming.as_slice());
            }
        }
    }

    pub fn into_module(self) -> Module<'ctx> {
        self.module
    }
}
