#![allow(unused_variables)]

fn main() {
    let mut g = safegraph::VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
    let e = (0_i64, 10_u32);
    safegraph::graph!(
        &mut g =>
        {(0, 1u32)} -- e --> {(0, 2u32)},
        {(0, 3u32)} -- e --> {(0, 4u32)}
    );
}
