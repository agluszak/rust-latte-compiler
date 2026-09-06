use crate::ir::Ir;

/// One explicit place defining what optimizations run, in what order.
pub(crate) fn optimize_program(program: &mut Ir) {
    for function in program.functions.values_mut() {
        crate::gvn::optimize(function);
    }
}
