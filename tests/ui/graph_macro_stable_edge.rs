fn main() {
    let mut g: safegraph::VecGraph<u32, u32> = Default::default();
    let e = 10u32;
    safegraph::graph!(&mut g => {1} -- e --> {2});
}
