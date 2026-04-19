#![allow(unused_variables)]

fn main() {
    let mut g = safegraph::VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
    safegraph::graph!(
        &mut g =>
        {(0, 1u32)} -- e {(0, 10u32)} --> {(0, 2u32)},
        {(0, 3u32)} -- e {(0, 11u32)} --> {(0, 4u32)}
    );
}
