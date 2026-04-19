use ui_test::custom_flags::rustfix::RustfixMode;
use ui_test::spanned::Spanned;
use ui_test::{dependencies::DependencyBuilder, run_tests, Config};

fn main() -> ui_test::color_eyre::Result<()> {
    let mut config = Config {
        output_conflict_handling: if std::env::var_os("BLESS").is_some() {
            ui_test::bless_output_files
        } else {
            ui_test::error_on_output_conflict
        },
        bless_command: Some("BLESS=1 cargo test --test ui".to_string()),
        ..Config::rustc("tests/ui")
    };

    // Match `trybuild`: compare whole-file `.stderr` snapshots rather than
    // requiring an inline `//~` annotation on every diagnostic, and don't apply
    // rustfix suggestions / expect `.fixed` files.
    config.comment_defaults.base().require_annotations = Spanned::dummy(false).into();
    config
        .comment_defaults
        .base()
        .set_custom("rustfix", RustfixMode::Disabled);

    // Build `safegraph` (and the `graph!` proc-macro it re-exports) and expose
    // it to the test files as the `safegraph` crate. We point the dependency
    // builder at a tiny helper crate (`tests/ui_dep`) whose only direct
    // dependency is `safegraph`, rather than at the root manifest: `ui_test`
    // panics on `safegraph`'s *optional* `sprs` dependency (pruned from a
    // default metadata resolve), and the helper crate keeps that transitive,
    // optional dep out of the set the harness inspects.
    config.comment_defaults.base().set_custom(
        "dependencies",
        DependencyBuilder {
            crate_manifest_path: "tests/ui_dep/Cargo.toml".into(),
            ..DependencyBuilder::default()
        },
    );

    // Type-name stability: strip the internal backend module path from monomorphized
    // type names so `…::linked_adj_edge::NodeRepr<_>` prints as just `NodeRepr<_>`,
    // matching how sibling types (`EdgeRepr`) already render and decoupling the
    // snapshots from the module layout.
    config.stderr_filter(r"safegraph::raw_graph::[a-z_]+::", "");

    // When a monomorphized type is long enough, rustc truncates it in the
    // diagnostic and writes the full name to a side file. That path is both
    // machine-specific (absolute) and run-specific (an unstable hash), so collapse
    // the whole quoted path to a placeholder.
    config.stderr_filter(r"'[^']*\.long-type-\d+\.txt'", "'$$LONG_TYPE_FILE'");

    run_tests(config)
}
