use crate::types::Type;

/// The five source-language builtins. Single source of truth for checking and codegen.
pub(crate) fn language_builtins() -> Vec<(&'static str, Vec<Type>, Type)> {
    vec![
        ("printInt", vec![Type::Int], Type::Void),
        ("printString", vec![Type::LatteString], Type::Void),
        ("error", vec![], Type::Void),
        ("readInt", vec![], Type::Int),
        ("readString", vec![], Type::LatteString),
    ]
}

pub(crate) fn is_language_builtin(name: &str) -> bool {
    language_builtins().iter().any(|(n, _, _)| *n == name)
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
    if is_language_builtin(source_name)
        || source_name == RUNTIME_NEW_STRING
        || source_name == RUNTIME_STRING_CONCAT
        || source_name == RUNTIME_STRING_EQUAL
    {
        source_name.to_string()
    } else {
        mangle_user(source_name)
    }
}
