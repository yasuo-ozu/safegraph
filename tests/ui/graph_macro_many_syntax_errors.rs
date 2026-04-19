use safegraph::VecGraph;

fn main() {
    // disallowed long-form empty forward edge
    let _a: VecGraph<u32, u32> = safegraph::graph!({1} -- --> {2});
    // disallowed long-form empty reverse edge
    let _b: VecGraph<u32, u32> = safegraph::graph!({1} <-- -- {2});
    // missing `>` in forward edge operator
    let _c: VecGraph<u32, u32> = safegraph::graph!({1} -- {2} -- {3});
    // missing destination node
    let _d: VecGraph<u32, u32> = safegraph::graph!({1} -->);
    // missing source node
    let _e: VecGraph<u32, u32> = safegraph::graph!(--> {1});
    // malformed reverse operator
    let _f: VecGraph<u32, u32> = safegraph::graph!({1} < -- {2});
    // malformed source node spec (empty braces)
    let _g: VecGraph<u32, u32> = safegraph::graph!({} --> {2});
    // malformed edge spec (empty braces)
    let _h: VecGraph<u32, u32> = safegraph::graph!({1} -- {} --> {2});
    // malformed destination node spec (empty braces)
    let _i: VecGraph<u32, u32> = safegraph::graph!({1} --> {});
    // missing comma between declarations
    let _j: VecGraph<u32, u32> = safegraph::graph!({1} --> {2} {3} --> {4});
    // invalid statement-mode prefix (`=>` without input graph expr)
    let mut base = VecGraph::<u32, u32>::default();
    let _k = &mut base;
    safegraph::graph!(=> {1} --> {2});
    // invalid statement-mode separator (`->` instead of `=>`)
    safegraph::graph!(&mut base -> {1} --> {2});
}
