use crate::ast::Ident;
use crate::ir::{Function, IrFunctionType, IrType, ValueBuilder, VariableId};
use crate::typechecker::Type;
use crate::typed_ast::TypedDecl;
use crate::{ast, ir};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Environment {
    variables: HashMap<ast::Ident, VariableId>,
    parent: Option<Box<Environment>>,
}

impl Environment {
    fn global() -> Self {
        Environment {
            variables: HashMap::new(),
            parent: None,
        }
    }

    fn local(&self) -> Self {
        Environment {
            variables: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    fn get_variable(&mut self, ident: &ast::Ident) -> Option<VariableId> {
        self.variables.get(ident).cloned()
    }

    fn add_variable(&mut self, ident: ast::Ident, id: VariableId) {
        self.variables.insert(ident, id);
    }
}

fn lower_type(ty: &Type) -> Option<IrType> {
    match ty {
        Type::Int => Some(IrType::Int),
        Type::Bool => Some(IrType::Bool),
        Type::LatteString => Some(IrType::String),
        Type::Void => None,
        Type::Function(_, _) => unimplemented!("Function types are not supported"),
    }
}

fn lower_fn_decl(decl: &TypedDecl) -> (String, ir::Function) {
    match decl {
        TypedDecl::Fn {
            return_type,
            name,
            args,
            body,
        } => {
            let return_ty = lower_type(return_type);
            let arg_names: Vec<Ident> = args
                .iter()
                .map(|arg| arg.value.name.value.clone())
                .collect();
            // FIXME
            let args: Vec<IrType> = args
                .into_iter()
                .map(|arg| lower_type(&arg.value.ty).expect("Void type in arguments"))
                .collect();
            let function_type = IrFunctionType {
                return_ty,
                args: args.clone(),
            };

            let mut function = ir::Function::new(function_type);
            let mut env = Environment::global();

            let start_block = function.new_block();
            function.seal_block(start_block);

            for (i, (name, ty)) in arg_names.into_iter().zip(args).enumerate() {
                let id = function.new_variable(ty);
                env.add_variable(name, id);
                function.write_variable(id, start_block, ValueBuilder::arg(i as u32));

                // FIXME
            }
            todo!()
        }
        _ => panic!("Expected function declaration"),
    }
}
