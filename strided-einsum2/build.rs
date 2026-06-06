fn main() {
    #[cfg(feature = "blas")]
    {
        println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
        println!("cargo:rerun-if-env-changed=MKLROOT");
        println!("cargo:rerun-if-env-changed=MKL_LIB_DIR");

        let explicit_providers = [
            cfg!(feature = "blas-accelerate"),
            cfg!(feature = "blas-openblas"),
            cfg!(feature = "blas-mkl"),
        ]
        .into_iter()
        .filter(|enabled| *enabled)
        .count();

        if explicit_providers > 1 {
            panic!("Select at most one explicit BLAS provider feature.");
        }

        if cfg!(feature = "blas-accelerate")
            || (explicit_providers == 0 && cfg!(target_os = "macos"))
        {
            link_accelerate();
        } else if cfg!(feature = "blas-mkl") {
            link_mkl_dynamic_parallel();
        } else {
            link_openblas();
        }
    }
}

#[cfg(feature = "blas")]
fn link_accelerate() {
    println!("cargo:rustc-link-lib=framework=Accelerate");
}

#[cfg(feature = "blas")]
fn link_openblas() {
    // Link against system OpenBLAS which provides CBLAS symbols.
    // On Ubuntu: apt install libopenblas-dev
    if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else if cfg!(target_os = "macos") {
        // Homebrew default paths on Apple Silicon and Intel Macs.
        println!("cargo:rustc-link-search=native=/opt/homebrew/opt/openblas/lib");
        println!("cargo:rustc-link-search=native=/usr/local/opt/openblas/lib");
    }
    println!("cargo:rustc-link-lib=openblas");
}

#[cfg(feature = "blas")]
fn link_mkl_dynamic_parallel() {
    if let Ok(lib_dir) = std::env::var("MKL_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else if let Ok(root) = std::env::var("MKLROOT") {
        println!("cargo:rustc-link-search=native={}/lib", root);
        println!("cargo:rustc-link-search=native={}/lib/intel64", root);
    }

    println!("cargo:rustc-link-lib=dylib=mkl_intel_lp64");
    println!("cargo:rustc-link-lib=dylib=mkl_intel_thread");
    println!("cargo:rustc-link-lib=dylib=mkl_core");
    println!("cargo:rustc-link-lib=dylib=iomp5");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }
}
