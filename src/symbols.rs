use crate::types::Type;

/// The five source-language builtins. Single source of truth for checking and codegen.
pub(crate) const LANGUAGE_BUILTINS: &[(&str, &[Type], Type)] = &[
    ("printInt", &[Type::Int], Type::Void),
    ("printString", &[Type::LatteString], Type::Void),
    ("error", &[], Type::Void),
    ("readInt", &[], Type::Int),
    ("readString", &[], Type::LatteString),
];

pub(crate) fn language_builtins() -> Vec<(&'static str, Vec<Type>, Type)> {
    LANGUAGE_BUILTINS
        .iter()
        .map(|(name, args, ret)| (*name, args.to_vec(), ret.clone()))
        .collect()
}

pub(crate) fn is_language_builtin(name: &str) -> bool {
    LANGUAGE_BUILTINS.iter().any(|(n, _, _)| *n == name)
}

/// Runtime helpers used by codegen. Not source-language builtins; their ABI
/// types are not all source-language types. Kept distinct on purpose.
pub(crate) const RUNTIME_NEW_STRING: &str = "newString";
pub(crate) const RUNTIME_STRING_CONCAT: &str = "stringConcat";
pub(crate) const RUNTIME_STRING_EQUAL: &str = "stringEqual";

/// User functions live in a namespace disjoint from builtins and runtime
/// helpers, apart from the required `main` entry point.
pub(crate) fn mangle_user(name: &str) -> String {
    if name == "main" {
        "main".to_string()
    } else {
        format!("latte_user_{name}")
    }
}

pub(crate) fn resolve_callee(source_name: &str) -> String {
    if is_language_builtin(source_name) {
        source_name.to_owned()
    } else {
        mangle_user(source_name)
    }
}
