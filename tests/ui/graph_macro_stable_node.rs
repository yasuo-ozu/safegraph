fn main() {
    let mut g: safegraph::VecGraph<u32, u32> = Default::default();
    let a = 1u32;
    let b = 2u32;
    safegraph::graph!(&mut g => a -- {10} --> b);
}
