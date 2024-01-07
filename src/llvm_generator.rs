use crate::ir::{BinaryOpCode, FunctionIr, Terminator, UnaryOpCode, Value, ValueId};
use crate::typechecker::{ReadyEnvironment, Type};
use either::Left;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicTypeEnum, FunctionType, StructType};
use inkwell::values::{BasicValue, BasicValueEnum};
use inkwell::AddressSpace;
use std::collections::BTreeMap;
use std::path::Path;

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

    pub fn declare(&self, name: &str, ir: &FunctionIr) {
        let fn_type = self.llvm_function_type(&ir.ty);
        self.module.add_function(name, fn_type, None);
    }

    fn get_rerouted(ir: &FunctionIr, id: ValueId) -> ValueId {
        let mut id = id;
        loop {
            match ir.ir.values[&id] {
                Value::Rerouted(new_id) => id = new_id,
                _ => return id,
            }
        }
    }

    pub fn generate(&self, name: &str, ir: &FunctionIr) {
        let function = self.module.get_function(name).unwrap();
        let mut basic_blocks = BTreeMap::new();
        let mut values: BTreeMap<ValueId, BasicValueEnum> = BTreeMap::new();
        let mut value_types: BTreeMap<ValueId, Type> = BTreeMap::new();
        for (id, ty) in &ir.ir.types {
            value_types.insert(*id, ty.clone());
        }

        for name in ir.ir.blocks.keys() {
            let this_block = self.context.append_basic_block(function, &name.to_string());
            basic_blocks.insert(name, this_block);
        }
        let mut phis = BTreeMap::new();
        for (name, block) in &ir.ir.blocks {
            let this_block = basic_blocks[name];
            self.builder.position_at_end(this_block);
            // First add phis
            for &id in &block.instructions {
                let value = &ir.ir.values[&id];
                let value_type = &ir.ir.types[&id];
                match value {
                    Value::Phi(phi) => {
                        // Incoming values will be set later
                        let llvm_phi = self
                            .builder
                            .build_phi(self.llvm_basic_type(value_type), &id.to_string())
                            .unwrap();
                        phis.insert(id, (phi, llvm_phi));

                        values.insert(id, llvm_phi.as_basic_value());
                    }
                    Value::Rerouted(rerouted) => {
                        let rerouted = Self::get_rerouted(ir, *rerouted);
                        values.insert(id, values[&rerouted]);
                    }
                    _ => {}
                }
            }

            // Then the rest
            for &id in &block.instructions {
                let value = ir.ir.values.get(&id).unwrap();
                match value {
                    Value::Int(i) => {
                        values.insert(
                            id,
                            self.context.i32_type().const_int(*i as u64, false).into(),
                        );
                        // TODO: 64 bit vs 32 bit
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
                            .build_bitcast(
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
                        values.insert(id, string_ptr.try_as_basic_value().unwrap_left());
                    }
                    Value::Bool(b) => {
                        values.insert(
                            id,
                            self.context.bool_type().const_int(*b as u64, false).into(),
                        );
                    }
                    Value::Call(var_id, args) => {
                        let name = &self.env.names[&var_id];
                        let function = self.module.get_function(name).unwrap();
                        let args = args
                            .iter()
                            .map(|arg| (*values.get(arg).unwrap()).into())
                            .collect::<Vec<_>>();
                        let value = self
                            .builder
                            .build_call(function, args.as_slice(), name)
                            .unwrap();
                        if let Left(value) = value.try_as_basic_value() {
                            values.insert(id, value);
                        }
                    }
                    Value::Argument(i) => {
                        values.insert(id, function.get_nth_param(*i).unwrap());
                    }
                    Value::BinaryOp(op, lhs, rhs) => {
                        match op {
                            BinaryOpCode::Add => {
                                if let Type::LatteString = value_types[&lhs] {
                                    let lhs = values[&lhs].into_pointer_value();
                                    let rhs = values[&rhs].into_pointer_value();
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
                                    values
                                        .insert(id, new_string.try_as_basic_value().unwrap_left());
                                } else if let Type::Int = value_types[&lhs] {
                                    let lhs = values[&lhs].into_int_value();
                                    let rhs = values[&rhs].into_int_value();
                                    values.insert(
                                        id,
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
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_int_sub(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            }
                            BinaryOpCode::Mul => {
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_int_mul(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            }
                            BinaryOpCode::Div => {
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_int_signed_div(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            }
                            BinaryOpCode::Mod => {
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_int_signed_rem(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            }
                            BinaryOpCode::Gt => {
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
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
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
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
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
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
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
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
                                // TODO: handle strings
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
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
                            BinaryOpCode::Neq => {
                                // TODO: handle strings
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
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
                            BinaryOpCode::And => {
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_and(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            }
                            BinaryOpCode::Or => {
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_or(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
                            }
                        }
                    }
                    Value::UnaryOp(op, val) => match op {
                        UnaryOpCode::Neg => {
                            let val = values.get(val).unwrap().into_int_value();
                            values.insert(
                                id,
                                self.builder
                                    .build_int_neg(val, &id.to_string())
                                    .unwrap()
                                    .into(),
                            );
                        }
                        UnaryOpCode::Not => {
                            let val = values.get(val).unwrap().into_int_value();
                            values.insert(
                                id,
                                self.builder.build_not(val, &id.to_string()).unwrap().into(),
                            );
                        }
                    },
                    Value::Phi(_) => {}
                    Value::Rerouted(_) => {}
                    Value::Undef => {}
                }
            }
            match block.terminator {
                Terminator::Return(val) => {
                    let val = values.get(&val).unwrap();
                    self.builder.build_return(Some(val)).unwrap();
                }
                Terminator::ReturnNoValue => {
                    self.builder.build_return(None).unwrap();
                }
                Terminator::Branch(val, then, else_) => {
                    let val = values[&val].into_int_value();
                    let then = basic_blocks[&then];
                    let else_ = basic_blocks.get(&else_).unwrap();
                    self.builder
                        .build_conditional_branch(val, then, *else_)
                        .unwrap();
                }
                Terminator::Jump(block) => {
                    let block = basic_blocks.get(&block).unwrap();
                    self.builder.build_unconditional_branch(*block).unwrap();
                }
            }
        }

        for (_id, (phi, llvm_phi)) in phis {
            let mut incoming = Vec::new();
            for (block, value) in phi.incoming() {
                incoming.push((&values[&value] as &dyn BasicValue, basic_blocks[&block]));
            }

            llvm_phi.add_incoming(incoming.as_slice());
        }
    }

    pub fn print(&self) {
        self.module.print_to_stderr();
    }

    pub fn compile_to_string(&self) -> String {
        self.module.print_to_string().to_string()
    }

    pub fn compile<P: AsRef<Path>>(&self, path: P) {
        self.module.print_to_file(path).unwrap();
    }
}
