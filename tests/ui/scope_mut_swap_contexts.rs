use core::ops::DerefMut;
use safegraph::BTreeGraph;

fn main() {
    let mut g1 = BTreeGraph::<u32, u32>::default();
    let mut g2 = BTreeGraph::<u32, u32>::default();
    // Force both graph borrows to `'static` so this UI test isolates
    // `'scope` incompatibility from shorter outer-borrow lifetime errors.
    let g1: &'static mut BTreeGraph<u32, u32> =
        unsafe { core::mem::transmute::<&mut BTreeGraph<u32, u32>, _>(&mut g1) };
    let g2: &'static mut BTreeGraph<u32, u32> =
        unsafe { core::mem::transmute::<&mut BTreeGraph<u32, u32>, _>(&mut g2) };
    g1.scope_mut(|mut wrap_a| {
        g2.scope_mut(|mut wrap_b| {
            core::mem::swap(wrap_a.deref_mut(), wrap_b.deref_mut());
        });
    });
}
