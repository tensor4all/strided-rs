fn main() {
    #[cfg(feature = "blas")]
    {
        // Link against system OpenBLAS which provides CBLAS symbols.
        // On macOS: brew install openblas
        // On Ubuntu: apt install libopenblas-dev
        if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        } else if cfg!(target_os = "macos") {
            // Homebrew default path on Apple Silicon
            println!("cargo:rustc-link-search=native=/opt/homebrew/opt/openblas/lib");
            // Homebrew default path on Intel
            println!("cargo:rustc-link-search=native=/usr/local/opt/openblas/lib");
        }
        println!("cargo:rustc-link-lib=openblas");
    }
}
