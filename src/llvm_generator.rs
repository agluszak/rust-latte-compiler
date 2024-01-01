use crate::ir::{BinaryOpCode, FunctionIr, Terminator, UnaryOpCode, Value, ValueId};
use crate::typechecker::{ReadyEnvironment, Type};
use either::Left;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicTypeEnum, FunctionType};
use inkwell::values::{BasicValue, BasicValueEnum};
use inkwell::AddressSpace;
use std::collections::BTreeMap;
use std::path::Path;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    env: ReadyEnvironment,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context, name: &str, env: ReadyEnvironment) -> Self {
        let module = context.create_module(name);
        let builder = context.create_builder();

        let codegen = CodeGen {
            context,
            module,
            builder,
            env,
        };
        codegen.add_builtins();

        codegen
    }

    pub fn add_builtins(&self) {
        let i32_type = self.context.i32_type();
        let void = self.context.void_type();
        let printf = self.module.add_function(
            "printf",
            i32_type.fn_type(&[i32_type.ptr_type(AddressSpace::default()).into()], true),
            Some(Linkage::External),
        );
        let print_int =
            self.module
                .add_function("printInt", void.fn_type(&[i32_type.into()], false), None);
        let basic_block = self.context.append_basic_block(print_int, "entry");
        self.builder.position_at_end(basic_block);
        // define void @printInt(i32 %x) {
        //     %t0 = getelementptr [4 x i8], [4 x i8]* @dnl, i32 0, i32 0
        //     call i32 (i8*, ...) @printf(i8* %t0, i32 %x)
        //     ret void
        // }
        let dnl = self.builder.build_global_string_ptr("%d\n", "dnl").unwrap();
        let x = print_int.get_nth_param(0).unwrap().into_int_value();
        self.builder
            .build_call(printf, &[dnl.as_pointer_value().into(), x.into()], "call")
            .unwrap();
        self.builder.build_return(None).unwrap();
    }

    fn llvm_basic_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Int => self.context.i32_type().into(),
            Type::Bool => self.context.bool_type().into(),
            Type::Void => panic!("void type is not a basic llvm type"),
            Type::Function(_, _) => panic!("function type is not a basic llvm type"),
            Type::LatteString => todo!("string type"),
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
                    Type::LatteString => todo!("string type"),
                }
            }
            _ => panic!("not a function type"),
        }
    }

    pub fn generate(&self, name: &str, ir: &FunctionIr) {
        let fn_type = self.llvm_function_type(&ir.ty);
        let function = self.module.add_function(name, fn_type, None);
        let mut basic_blocks = BTreeMap::new();
        let mut values: BTreeMap<ValueId, BasicValueEnum> = BTreeMap::new();
        for name in ir.ir.blocks.keys() {
            let this_block = self.context.append_basic_block(function, &name.to_string());
            basic_blocks.insert(name, this_block);
        }
        let mut phis = BTreeMap::new();
        for (name, block) in &ir.ir.blocks {
            let this_block = basic_blocks[name];
            self.builder.position_at_end(this_block);
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
                    Value::String(_) => todo!("string"),
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
                                let lhs = values.get(lhs).unwrap().into_int_value();
                                let rhs = values.get(rhs).unwrap().into_int_value();
                                values.insert(
                                    id,
                                    self.builder
                                        .build_int_add(lhs, rhs, &id.to_string())
                                        .unwrap()
                                        .into(),
                                );
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
                    Value::Phi(phi) => {
                        // Incoming values will be set later
                        let llvm_phi = self
                            .builder
                            .build_phi(self.context.i32_type(), &id.to_string())
                            .unwrap();
                        phis.insert(id, (phi, llvm_phi));

                        values.insert(id, llvm_phi.as_basic_value());
                    }
                    Value::Rerouted(rerouted) => {
                        values.insert(id, values[&rerouted]);
                    }
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
                    let val = values.get(&val).unwrap().into_int_value();
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
