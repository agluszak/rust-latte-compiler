use crate::ir::Ir;

/// One explicit place defining what optimizations run, in what order.
pub(crate) fn optimize_program(program: &mut Ir) {
    for function in program.functions.values_mut() {
        #[cfg(debug_assertions)]
        crate::verify::verify(function).expect("IR must be valid before GVN");
        crate::gvn::optimize(function);
        #[cfg(debug_assertions)]
        crate::verify::verify(function).expect("IR must be valid after GVN");
    }
}
