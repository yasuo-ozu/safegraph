use safegraph::graph::GraphOperation;

fn main() {
    let mut g = safegraph::VecGraph::<(i64, u32), (i64, u32)>::default().stabilize();
    safegraph::graph!(
        &mut g =>
        a {(0, 1u32)} -- {(0, 10u32)} --> b {(0, 2u32)},
        a {(0, 1u32)} -- {(0, 11u32)} --> c {(0, 3u32)}
    );
}
