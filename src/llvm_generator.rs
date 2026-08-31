use crate::ir::{BinaryOpCode, BlockId, FunctionIr, Phi, Terminator, UnaryOpCode, Value};
use crate::typechecker::{ReadyEnvironment, Type};
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicTypeEnum, FunctionType, StructType};
use inkwell::values::{BasicValue, BasicValueEnum, PhiValue};

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    string_type: StructType<'ctx>,
    env: ReadyEnvironment,
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
        };
        codegen.declare_builtins();

        codegen
    }

    pub fn declare_builtins(&self) {
        let i8_type = self.context.i8_type();
        let i32_type = self.context.i32_type();
        let void = self.context.void_type();

        self.module.add_function(
            "printInt",
            void.fn_type(&[i32_type.into()], false),
            Some(Linkage::External),
        );

        self.module.add_function(
            "printString",
            void.fn_type(
                &[self.string_type.ptr_type(AddressSpace::default()).into()],
                false,
            ),
            Some(Linkage::External),
        );

        self.module.add_function(
            "readInt",
            i32_type.fn_type(&[], false),
            Some(Linkage::External),
        );

        self.module.add_function(
            "readString",
            self.string_type
                .ptr_type(AddressSpace::default())
                .fn_type(&[], false),
            Some(Linkage::External),
        );

        self.module
            .add_function("error", void.fn_type(&[], false), Some(Linkage::External));

        self.module.add_function(
            "newString",
            self.string_type.ptr_type(AddressSpace::default()).fn_type(
                &[
                    i8_type.ptr_type(AddressSpace::default()).into(),
                    i32_type.into(),
                ],
                false,
            ),
            Some(Linkage::External),
        );

        self.module.add_function(
            "stringConcat",
            self.string_type.ptr_type(AddressSpace::default()).fn_type(
                &[
                    self.string_type.ptr_type(AddressSpace::default()).into(),
                    self.string_type.ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            Some(Linkage::External),
        );

        // TODO: boolean
        self.module.add_function(
            "stringEqual",
            i32_type.fn_type(
                &[
                    self.string_type.ptr_type(AddressSpace::default()).into(),
                    self.string_type.ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
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
                    Type::Bool => self.context.bool_type().fn_type(&args, false),
                    Type::Int => self.context.i32_type().fn_type(&args, false),
                    Type::Function(_, _) => panic!("function type is not a basic llvm type"),
                    Type::LatteString => self
                        .string_type
                        .ptr_type(AddressSpace::default())
                        .fn_type(&args, false),
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

    pub fn declare(&self, name: &str, ir: &FunctionIr) {
        let fn_type = self.llvm_function_type(&ir.ty);
        self.module.add_function(name, fn_type, None);
    }

    pub fn generate(&self, name: &str, ir: &FunctionIr) {
        let function = self.module.get_function(name).unwrap();
        let basic_blocks: Vec<BasicBlock> = (0..ir.blocks.len())
            .map(|index| {
                let name = BlockId::from_index(index);
                self.context.append_basic_block(function, &name.to_string())
            })
            .collect();
        let mut values: Vec<Option<BasicValueEnum>> = vec![None; ir.values.len()];
        let mut phis: Vec<Option<(Phi, PhiValue)>> = vec![None; ir.values.len()];
        for (block_index, block) in ir.blocks.iter().enumerate() {
            let this_block = basic_blocks[block_index];
            self.builder.position_at_end(this_block);
            // Phis come first, so their values are available to all instructions.
            for &id in &block.phis {
                let value_data = &ir.values[id.index()];
                let Value::Phi(phi) = &value_data.kind else {
                    unreachable!("block.phis must only contain phi values");
                };
                // Incoming values will be set later
                let llvm_phi = self
                    .builder
                    .build_phi(self.llvm_basic_type(&value_data.ty), &id.to_string())
                    .unwrap();
                phis[id.index()] = Some((phi.clone(), llvm_phi));

                values[id.index()] = Some(llvm_phi.as_basic_value());
            }

            // Then the rest
            for &id in &block.instructions {
                let value_data = &ir.values[id.index()];
                let value = &value_data.kind;
                match value {
                    Value::Int(i) => {
                        values[id.index()] =
                            Some(self.context.i32_type().const_int(*i as u64, true).into());
                    }
                    Value::String(s) => {
                        // TODO: fix leak
                        let len = self.context.i32_type().const_int(s.len() as u64, false);
                        let const_str = self.context.const_string(s.as_bytes(), false);
                        let str_ptr = self
                            .builder
                            .build_alloca(const_str.get_type(), "str_ptr")
                            .unwrap();
                        self.builder.build_store(str_ptr, const_str).unwrap();
                        let str_ptr = self
                            .builder
                            .build_bit_cast(
                                str_ptr,
                                self.context.i8_type().ptr_type(AddressSpace::default()),
                                "str_ptr",
                            )
                            .unwrap();
                        let new_string_fn = self.module.get_function("newString").unwrap();
                        let string_ptr = self
                            .builder
                            .build_call(new_string_fn, &[str_ptr.into(), len.into()], "new_string")
                            .unwrap();
                        values[id.index()] = Some(string_ptr.try_as_basic_value().unwrap_basic());
                    }
                    Value::Bool(b) => {
                        values[id.index()] =
                            Some(self.context.bool_type().const_int(*b as u64, false).into());
                    }
                    Value::Call(var_id, args) => {
                        let name = &self.env.names[var_id];
                        let function = self.module.get_function(name).unwrap();
                        let args = args
                            .iter()
                            .map(|arg| values[arg.index()].unwrap().into())
                            .collect::<Vec<_>>();
                        let value = self
                            .builder
                            .build_call(function, args.as_slice(), name)
                            .unwrap();
                        if let Some(value) = value.try_as_basic_value().basic() {
                            values[id.index()] = Some(value);
                        }
                    }
                    Value::Argument(i) => {
                        values[id.index()] = Some(function.get_nth_param(*i).unwrap());
                    }
                    Value::BinaryOp(op, lhs, rhs) => match op {
                        BinaryOpCode::Add => {
                            if let Type::LatteString = ir.values[lhs.index()].ty {
                                let lhs = values[lhs.index()].unwrap().into_pointer_value();
                                let rhs = values[rhs.index()].unwrap().into_pointer_value();
                                let string_concat_fn =
                                    self.module.get_function("stringConcat").unwrap();
                                let new_string = self
                                    .builder
                                    .build_call(
                                        string_concat_fn,
                                        &[lhs.into(), rhs.into()],
                                        "new_string",
                                    )
                                    .unwrap();
                                values[id.index()] =
                                    Some(new_string.try_as_basic_value().unwrap_basic());
                            } else if let Type::Int = ir.values[lhs.index()].ty {
                                let lhs = values[lhs.index()].unwrap().into_int_value();
                                let rhs = values[rhs.index()].unwrap().into_int_value();
                                values[id.index()] = Some(
                                    self.builder
                                        .build_int_add(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            } else {
                                panic!("invalid type for add");
                            }
                        }
                        BinaryOpCode::Sub => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_sub(lhs, rhs, &id.to_string())
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Mul => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_mul(lhs, rhs, &id.to_string())
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Div => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_signed_div(lhs, rhs, &id.to_string())
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Mod => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_signed_rem(lhs, rhs, &id.to_string())
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Gt => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_compare(
                                        inkwell::IntPredicate::SGT,
                                        lhs,
                                        rhs,
                                        &id.to_string(),
                                    )
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Lt => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_compare(
                                        inkwell::IntPredicate::SLT,
                                        lhs,
                                        rhs,
                                        &id.to_string(),
                                    )
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Gte => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_compare(
                                        inkwell::IntPredicate::SGE,
                                        lhs,
                                        rhs,
                                        &id.to_string(),
                                    )
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Lte => {
                            let lhs = values[lhs.index()].unwrap().into_int_value();
                            let rhs = values[rhs.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_compare(
                                        inkwell::IntPredicate::SLE,
                                        lhs,
                                        rhs,
                                        &id.to_string(),
                                    )
                                    .unwrap()
                                    .into(),
                            );
                        }
                        BinaryOpCode::Eq => {
                            if let Type::LatteString = ir.values[lhs.index()].ty {
                                let lhs = values[lhs.index()].unwrap().into_pointer_value();
                                let rhs = values[rhs.index()].unwrap().into_pointer_value();
                                let string_equal = self
                                    .builder
                                    .build_call(
                                        self.module.get_function("stringEqual").unwrap(),
                                        &[lhs.into(), rhs.into()],
                                        "strings_equal",
                                    )
                                    .unwrap()
                                    .try_as_basic_value()
                                    .unwrap_basic()
                                    .into_int_value();
                                values[id.index()] = Some(
                                    self.builder
                                        .build_int_compare(
                                            inkwell::IntPredicate::NE,
                                            string_equal,
                                            string_equal.get_type().const_zero(),
                                            &id.to_string(),
                                        )
                                        .unwrap()
                                        .into(),
                                );
                            } else {
                                let lhs = values[lhs.index()].unwrap().into_int_value();
                                let rhs = values[rhs.index()].unwrap().into_int_value();
                                values[id.index()] = Some(
                                    self.builder
                                        .build_int_compare(
                                            inkwell::IntPredicate::EQ,
                                            lhs,
                                            rhs,
                                            &id.to_string(),
                                        )
                                        .unwrap()
                                        .into(),
                                );
                            }
                        }
                        BinaryOpCode::Neq => {
                            if let Type::LatteString = ir.values[lhs.index()].ty {
                                let lhs = values[lhs.index()].unwrap().into_pointer_value();
                                let rhs = values[rhs.index()].unwrap().into_pointer_value();
                                let string_equal = self
                                    .builder
                                    .build_call(
                                        self.module.get_function("stringEqual").unwrap(),
                                        &[lhs.into(), rhs.into()],
                                        "strings_equal",
                                    )
                                    .unwrap()
                                    .try_as_basic_value()
                                    .unwrap_basic()
                                    .into_int_value();
                                values[id.index()] = Some(
                                    self.builder
                                        .build_int_compare(
                                            inkwell::IntPredicate::EQ,
                                            string_equal,
                                            string_equal.get_type().const_zero(),
                                            &id.to_string(),
                                        )
                                        .unwrap()
                                        .into(),
                                );
                            } else {
                                let lhs = values[lhs.index()].unwrap().into_int_value();
                                let rhs = values[rhs.index()].unwrap().into_int_value();
                                values[id.index()] = Some(
                                    self.builder
                                        .build_int_compare(
                                            inkwell::IntPredicate::NE,
                                            lhs,
                                            rhs,
                                            &id.to_string(),
                                        )
                                        .unwrap()
                                        .into(),
                                );
                            }
                        }
                    },
                    Value::UnaryOp(op, val) => match op {
                        UnaryOpCode::Neg => {
                            let val = values[val.index()].unwrap().into_int_value();
                            values[id.index()] = Some(
                                self.builder
                                    .build_int_neg(val, &id.to_string())
                                    .unwrap()
                                    .into(),
                            );
                        }
                        UnaryOpCode::Not => {
                            let val = values[val.index()].unwrap().into_int_value();
                            values[id.index()] =
                                Some(self.builder.build_not(val, &id.to_string()).unwrap().into());
                        }
                    },
                    Value::Phi(_) => unreachable!("phi values live in block.phis"),
                    Value::Undef => {
                        values[id.index()] = Some(self.llvm_undef(&value_data.ty));
                    }
                }
            }
            match block.terminator {
                Terminator::Return(val) => {
                    let val = values[val.index()].unwrap();
                    self.builder.build_return(Some(&val)).unwrap();
                }
                Terminator::ReturnNoValue => {
                    self.builder.build_return(None).unwrap();
                }
                Terminator::Branch(val, then, else_) => {
                    let val = values[val.index()].unwrap().into_int_value();
                    let then = basic_blocks[then.index()];
                    let else_ = basic_blocks[else_.index()];
                    self.builder
                        .build_conditional_branch(val, then, else_)
                        .unwrap();
                }
                Terminator::Jump(block) => {
                    let block = basic_blocks[block.index()];
                    self.builder.build_unconditional_branch(block).unwrap();
                }
            }
        }

        for (phi, llvm_phi) in phis.into_iter().flatten() {
            let mut incoming: Vec<(BasicValueEnum, BasicBlock)> = Vec::new();
            for (block, value) in &phi.incoming {
                let value = values[value.index()].unwrap();
                incoming.push((value, basic_blocks[block.index()]));
            }
            let incoming: Vec<(&dyn BasicValue, BasicBlock)> = incoming
                .iter()
                .map(|(value, block)| (value as &dyn BasicValue, *block))
                .collect();

            llvm_phi.add_incoming(incoming.as_slice());
        }
    }

    pub fn into_module(self) -> Module<'ctx> {
        self.module
    }
}
