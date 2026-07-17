#[test]
fn workspace_faer_dependency_uses_only_dense_runtime_features() {
    let manifest = include_str!("../../Cargo.toml");
    let faer_line = manifest
        .lines()
        .find(|line| line.starts_with("faer = "))
        .expect("workspace must declare faer");

    assert!(
        faer_line.contains("default-features = false"),
        "faer defaults pull sparse, npy, and rand dependencies: {faer_line}"
    );
    assert!(faer_line.contains("features = [\"std\", \"rayon\"]"));
}
