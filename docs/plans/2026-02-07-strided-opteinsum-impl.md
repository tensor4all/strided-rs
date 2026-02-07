# strided-opteinsum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port mdarray-einsum to strided-rs as a new `strided-opteinsum` crate with nested string notation, mixed f64/c64 types, and omeco fallback optimization.

**Architecture:** Parse nested einsum strings like `"(ij,jk),kl->il"` into a contraction tree (`EinsumExpr`). Evaluate recursively: single-tensor ops use diagonal stride trick + reduce, pairwise ops delegate to `einsum2_into`, 3+ tensor nodes fall back to omeco. Mixed f64/c64 handled via `EinsumOperand` enum with automatic promotion.

**Tech Stack:** Rust, strided-view, strided-kernel, strided-einsum2, omeco 0.2, num-complex, thiserror

**Reference files:**
- Design: `docs/plans/2026-02-07-strided-opteinsum-design.md`
- mdarray-einsum source: `../tensor4all-rs/crates/mdarray-einsum/src/`
- strided-einsum2: `strided-einsum2/src/`
- OMEinsum.jl: `~/git/OMEinsum.jl/src/`

---

### Task 1: Add `diagonal_view` to strided-view

**Files:**
- Modify: `strided-view/src/view.rs`
- Test: `strided-view/src/view.rs` (inline tests)

**Step 1: Write failing tests**

Add to the existing `#[cfg(test)] mod tests` in `strided-view/src/view.rs`:

```rust
#[test]
fn test_diagonal_view_2d() {
    // A[i,i] shape=[3,3] row-major strides=[3,1]
    // diagonal: shape=[3] strides=[4] (3+1)
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[3, 3], &[3, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 1)]).unwrap();
    assert_eq!(diag.dims(), &[3]);
    assert_eq!(diag.strides(), &[4]);
    assert_eq!(diag.get(&[0]), 0.0); // A[0,0]
    assert_eq!(diag.get(&[1]), 4.0); // A[1,1]
    assert_eq!(diag.get(&[2]), 8.0); // A[2,2]
}

#[test]
fn test_diagonal_view_3d_adjacent() {
    // A[i,i,j] shape=[2,2,3] row-major strides=[6,3,1]
    // diagonal over (0,1): shape=[2,3] strides=[9,1] (6+3)
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 2, 3], &[6, 3, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 1)]).unwrap();
    assert_eq!(diag.dims(), &[2, 3]);
    assert_eq!(diag.strides(), &[9, 1]);
    // A[0,0,:] = [0,1,2], A[1,1,:] = [9,10,11]
    assert_eq!(diag.get(&[0, 0]), 0.0);
    assert_eq!(diag.get(&[0, 2]), 2.0);
    assert_eq!(diag.get(&[1, 0]), 9.0);
}

#[test]
fn test_diagonal_view_3d_non_adjacent() {
    // A[i,j,i] shape=[2,3,2] row-major strides=[6,2,1]
    // diagonal over (0,2): shape=[2,3] strides=[7,2] (6+1)
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 3, 2], &[6, 2, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 2)]).unwrap();
    assert_eq!(diag.dims(), &[2, 3]);
    assert_eq!(diag.strides(), &[7, 2]);
    // A[0,:,0] = [0,2,4], A[1,:,1] = [7,9,11]
    assert_eq!(diag.get(&[0, 0]), 0.0);
    assert_eq!(diag.get(&[0, 1]), 2.0);
    assert_eq!(diag.get(&[1, 0]), 7.0);
    assert_eq!(diag.get(&[1, 2]), 11.0);
}

#[test]
fn test_diagonal_view_two_pairs() {
    // A[i,j,i,j] shape=[2,3,2,3] -> A_diag[i,j] shape=[2,3]
    // strides: row-major [18,6,3,1]
    // pair (0,2): fuse to stride 18+3=21, remove axis 2
    // pair (1,3): fuse to stride 6+1=7, remove axis 3
    // result: shape=[2,3] strides=[21,7]
    let data: Vec<f64> = (0..36).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 3, 2, 3], &[18, 6, 3, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 2), (1, 3)]).unwrap();
    assert_eq!(diag.dims(), &[2, 3]);
    assert_eq!(diag.strides(), &[21, 7]);
    // A[0,0,0,0] = 0, A[1,1,1,1] = 18+6+3+1 = 28
    assert_eq!(diag.get(&[0, 0]), 0.0);
    assert_eq!(diag.get(&[1, 1]), 28.0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p strided-view test_diagonal_view`
Expected: Compile error — `diagonal_view` method does not exist

**Step 3: Implement `diagonal_view`**

Add to `impl<'a, T, Op: ElementOp> StridedView<'a, T, Op>` in `strided-view/src/view.rs`:

```rust
/// Create a diagonal view by fusing repeated axis pairs.
///
/// For each pair `(a, b)`, the two axes are merged into one:
/// - New stride = `strides[a] + strides[b]`
/// - New dim = `min(dims[a], dims[b])`
/// - The higher-numbered axis is removed.
///
/// Pairs are processed in order. Axis indices refer to the
/// **original** (pre-removal) axis numbering.
///
/// # Example
/// ```text
/// A[i,i,j] shape=[n,n,m] strides=[s0,s1,s2]
///   -> diagonal_view(&[(0,1)])
///   -> shape=[n,m] strides=[s0+s1, s2]
/// ```
pub fn diagonal_view(&self, axis_pairs: &[(usize, usize)]) -> Result<StridedView<'a, T, Op>> {
    let ndim = self.ndim();
    let mut dims: Vec<usize> = self.dims().to_vec();
    let mut strides: Vec<isize> = self.strides().to_vec();

    // Collect axes to remove (the higher axis of each pair)
    // and fuse strides into the lower axis.
    // We process in original axis numbering, then remove in reverse order.
    let mut axes_to_remove = Vec::new();
    for &(a, b) in axis_pairs {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if lo >= ndim || hi >= ndim {
            return Err(StridedError::InvalidAxis { axis: hi, rank: ndim });
        }
        if lo == hi {
            return Err(StridedError::InvalidAxis { axis: lo, rank: ndim });
        }
        // Fuse: sum strides, take min dim
        strides[lo] = strides[lo] + strides[hi];
        dims[lo] = dims[lo].min(dims[hi]);
        axes_to_remove.push(hi);
    }

    // Remove axes in descending order to preserve indices
    axes_to_remove.sort_unstable();
    axes_to_remove.dedup();
    for &ax in axes_to_remove.iter().rev() {
        dims.remove(ax);
        strides.remove(ax);
    }

    // Safety: same data, same offset, just different shape/strides
    // The diagonal elements are a subset of the original data
    unsafe { Ok(StridedView::new_unchecked(self.data(), &dims, &strides, self.offset())) }
}
```

**Step 4: Run tests**

Run: `cargo test -p strided-view test_diagonal_view`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add strided-view/src/view.rs
git commit -m "feat(strided-view): add diagonal_view for zero-copy diagonal extraction"
```

---

### Task 2: Scaffold strided-opteinsum crate

**Files:**
- Create: `strided-opteinsum/Cargo.toml`
- Create: `strided-opteinsum/src/lib.rs`
- Modify: `Cargo.toml` (workspace)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "strided-opteinsum"
version = "0.1.0"
edition = "2021"

[dependencies]
strided-view = { path = "../strided-view" }
strided-kernel = { path = "../strided-kernel" }
strided-einsum2 = { path = "../strided-einsum2" }
omeco = "0.2"
num-complex = "0.4"
num-traits = "0.2"
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
```

**Step 2: Create src/lib.rs**

```rust
pub mod error;
pub mod operand;
pub mod parse;
pub mod expr;
pub mod single_tensor;
pub mod typed_tensor;

pub use error::{EinsumError, Result};
pub use operand::{EinsumOperand, StridedData};
pub use expr::EinsumCode;
pub use parse::parse_einsum;
pub use typed_tensor::TypedTensor;
```

**Step 3: Create stub modules**

Create each of these as minimal files so the crate compiles:

`strided-opteinsum/src/error.rs`:
```rust
#[derive(Debug, thiserror::Error)]
pub enum EinsumError {
    #[error("parse error: {0}")]
    ParseError(String),

    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),

    #[error(transparent)]
    Einsum2(#[from] strided_einsum2::EinsumError),

    #[error("dimension mismatch for axis '{axis}': {dim_a} vs {dim_b}")]
    DimensionMismatch {
        axis: String,
        dim_a: usize,
        dim_b: usize,
    },

    #[error("output axis '{0}' not found in any input")]
    OrphanOutputAxis(String),
}

pub type Result<T> = std::result::Result<T, EinsumError>;
```

`strided-opteinsum/src/operand.rs`:
```rust
// Stub — implemented in Task 3
```

`strided-opteinsum/src/parse.rs`:
```rust
// Stub — implemented in Task 4
```

`strided-opteinsum/src/expr.rs`:
```rust
// Stub — implemented in Task 5
```

`strided-opteinsum/src/single_tensor.rs`:
```rust
// Stub — implemented in Task 6
```

`strided-opteinsum/src/typed_tensor.rs`:
```rust
// Stub — implemented in Task 8
```

**Step 4: Add to workspace**

In root `Cargo.toml`, add `"strided-opteinsum"` to members:
```toml
[workspace]
members = ["strided-view", "strided-kernel", "strided-einsum2", "strided-opteinsum"]
resolver = "2"
```

**Step 5: Verify build**

Run: `cargo check -p strided-opteinsum`
Expected: Compiles (with unused warnings)

**Step 6: Commit**

```bash
git add strided-opteinsum/ Cargo.toml
git commit -m "feat: scaffold strided-opteinsum crate with stub modules"
```

---

### Task 3: Implement EinsumOperand and StridedData

**Files:**
- Modify: `strided-opteinsum/src/operand.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use strided_view::StridedArray;

    #[test]
    fn test_f64_owned() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let op = EinsumOperand::from(arr);
        assert!(op.is_f64());
        assert!(!op.is_c64());
        assert_eq!(op.dims(), &[2, 3]);
    }

    #[test]
    fn test_c64_owned() {
        let arr = StridedArray::<Complex64>::col_major(&[4, 5]);
        let op = EinsumOperand::from(arr);
        assert!(op.is_c64());
        assert_eq!(op.dims(), &[4, 5]);
    }

    #[test]
    fn test_f64_view() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let view = arr.view();
        let op = EinsumOperand::from_view_f64(&view);
        assert!(op.is_f64());
        assert_eq!(op.dims(), &[2, 3]);
    }

    #[test]
    fn test_promote_f64_to_c64() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 2]);
        arr.data_mut()[0] = 1.0;
        arr.data_mut()[1] = 2.0;
        let op = EinsumOperand::from(arr);
        let promoted = op.to_c64_owned();
        assert!(promoted.is_c64());
        match &promoted {
            EinsumOperand::C64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.data()[0], Complex64::new(1.0, 0.0));
                assert_eq!(arr.data()[1], Complex64::new(2.0, 0.0));
            }
            _ => panic!("expected C64"),
        }
    }
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p strided-opteinsum test_f64_owned`
Expected: Compile error

**Step 3: Implement**

```rust
use num_complex::Complex64;
use num_traits::Zero;
use strided_view::{StridedArray, StridedView};

/// Heterogeneous operand: each input carries its own scalar type and ownership.
pub enum EinsumOperand<'a> {
    F64(StridedData<'a, f64>),
    C64(StridedData<'a, Complex64>),
}

/// Owned or borrowed strided data.
pub enum StridedData<'a, T> {
    Owned(StridedArray<T>),
    View(StridedView<'a, T>),
}

impl<'a, T> StridedData<'a, T> {
    pub fn dims(&self) -> &[usize] {
        match self {
            StridedData::Owned(a) => a.dims(),
            StridedData::View(v) => v.dims(),
        }
    }

    pub fn as_view(&self) -> StridedView<'_, T> {
        match self {
            StridedData::Owned(a) => a.view(),
            StridedData::View(v) => v.clone(),
        }
    }

    pub fn as_array(&self) -> &StridedArray<T> {
        match self {
            StridedData::Owned(a) => a,
            StridedData::View(_) => panic!("expected Owned, got View"),
        }
    }
}

impl<'a> EinsumOperand<'a> {
    pub fn is_f64(&self) -> bool {
        matches!(self, EinsumOperand::F64(_))
    }

    pub fn is_c64(&self) -> bool {
        matches!(self, EinsumOperand::C64(_))
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            EinsumOperand::F64(d) => d.dims(),
            EinsumOperand::C64(d) => d.dims(),
        }
    }

    pub fn from_view_f64(view: &StridedView<'a, f64>) -> Self {
        EinsumOperand::F64(StridedData::View(view.clone()))
    }

    pub fn from_view_c64(view: &StridedView<'a, Complex64>) -> Self {
        EinsumOperand::C64(StridedData::View(view.clone()))
    }

    /// Promote to C64-owned. If already C64, extracts or copies.
    /// If F64, converts each element to Complex64.
    pub fn to_c64_owned(self) -> EinsumOperand<'static> {
        match self {
            EinsumOperand::C64(StridedData::Owned(a)) => {
                EinsumOperand::C64(StridedData::Owned(a))
            }
            EinsumOperand::C64(StridedData::View(v)) => {
                let mut out = StridedArray::<Complex64>::col_major(v.dims());
                strided_kernel::copy_into(&mut out.view_mut(), &v).unwrap();
                EinsumOperand::C64(StridedData::Owned(out))
            }
            EinsumOperand::F64(StridedData::Owned(a)) => {
                let c64_data: Vec<Complex64> =
                    a.data().iter().map(|&x| Complex64::new(x, 0.0)).collect();
                let arr = StridedArray::from_parts(
                    c64_data,
                    a.dims(),
                    a.strides(),
                    0,
                ).unwrap();
                EinsumOperand::C64(StridedData::Owned(arr))
            }
            EinsumOperand::F64(StridedData::View(v)) => {
                let mut tmp = StridedArray::<f64>::col_major(v.dims());
                strided_kernel::copy_into(&mut tmp.view_mut(), &v).unwrap();
                let c64_data: Vec<Complex64> =
                    tmp.data().iter().map(|&x| Complex64::new(x, 0.0)).collect();
                let arr = StridedArray::from_parts(
                    c64_data,
                    tmp.dims(),
                    tmp.strides(),
                    0,
                ).unwrap();
                EinsumOperand::C64(StridedData::Owned(arr))
            }
        }
    }
}

impl From<StridedArray<f64>> for EinsumOperand<'static> {
    fn from(a: StridedArray<f64>) -> Self {
        EinsumOperand::F64(StridedData::Owned(a))
    }
}

impl From<StridedArray<Complex64>> for EinsumOperand<'static> {
    fn from(a: StridedArray<Complex64>) -> Self {
        EinsumOperand::C64(StridedData::Owned(a))
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p strided-opteinsum operand`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add strided-opteinsum/src/operand.rs
git commit -m "feat(opteinsum): implement EinsumOperand with F64/C64 and Owned/View"
```

---

### Task 4: Implement nested string parser

**Files:**
- Modify: `strided-opteinsum/src/parse.rs`

The parser handles OMEinsum-compatible notation:
- `"ij,jk->ik"` — flat (no nesting)
- `"(ij,jk),kl->il"` — nested
- `"((ij,jk),(kl,lm))->im"` — deeply nested

**Step 1: Define parsed types and write failing tests**

```rust
use crate::error::{EinsumError, Result};

/// Parsed contraction tree node (no operands yet, just structure).
#[derive(Debug, Clone, PartialEq)]
pub enum EinsumNode {
    /// Leaf: indices for a single tensor, with 0-based tensor index.
    Leaf { ids: Vec<char>, tensor_index: usize },
    /// Contraction of children.
    Contract { args: Vec<EinsumNode> },
}

/// Parsed einsum code: contraction tree + final output indices.
#[derive(Debug, Clone, PartialEq)]
pub struct EinsumCode {
    pub root: EinsumNode,
    pub output_ids: Vec<char>,
}

/// Parse an einsum string like "(ij,jk),kl->il" into an EinsumCode.
pub fn parse_einsum(s: &str) -> Result<EinsumCode> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_flat() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'k']);
        // Flat: single Contract with two leaves
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], EinsumNode::Leaf { ids: vec!['i', 'j'], tensor_index: 0 });
                assert_eq!(args[1], EinsumNode::Leaf { ids: vec!['j', 'k'], tensor_index: 1 });
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_nested() {
        let code = parse_einsum("(ij,jk),kl->il").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'l']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                // First arg is nested Contract
                match &args[0] {
                    EinsumNode::Contract { args: inner } => {
                        assert_eq!(inner.len(), 2);
                        assert_eq!(inner[0], EinsumNode::Leaf { ids: vec!['i', 'j'], tensor_index: 0 });
                        assert_eq!(inner[1], EinsumNode::Leaf { ids: vec!['j', 'k'], tensor_index: 1 });
                    }
                    _ => panic!("expected inner Contract"),
                }
                assert_eq!(args[1], EinsumNode::Leaf { ids: vec!['k', 'l'], tensor_index: 2 });
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_deep_nested() {
        let code = parse_einsum("((ij,jk),(kl,lm))->im").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'm']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                // Both children are nested Contracts
                match &args[0] {
                    EinsumNode::Contract { args: left } => {
                        assert_eq!(left.len(), 2);
                        assert_eq!(left[0], EinsumNode::Leaf { ids: vec!['i', 'j'], tensor_index: 0 });
                        assert_eq!(left[1], EinsumNode::Leaf { ids: vec!['j', 'k'], tensor_index: 1 });
                    }
                    _ => panic!("expected left Contract"),
                }
                match &args[1] {
                    EinsumNode::Contract { args: right } => {
                        assert_eq!(right.len(), 2);
                        assert_eq!(right[0], EinsumNode::Leaf { ids: vec!['k', 'l'], tensor_index: 2 });
                        assert_eq!(right[1], EinsumNode::Leaf { ids: vec!['l', 'm'], tensor_index: 3 });
                    }
                    _ => panic!("expected right Contract"),
                }
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_scalar_output() {
        let code = parse_einsum("ij,ji->").unwrap();
        assert_eq!(code.output_ids, vec![]);
    }

    #[test]
    fn test_parse_single_tensor() {
        let code = parse_einsum("ijk->kji").unwrap();
        assert_eq!(code.output_ids, vec!['k', 'j', 'i']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], EinsumNode::Leaf { ids: vec!['i', 'j', 'k'], tensor_index: 0 });
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_three_flat() {
        // 3 flat tensors — will trigger omeco fallback at evaluation time
        let code = parse_einsum("ij,jk,kl->il").unwrap();
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 3);
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_whitespace() {
        let code = parse_einsum(" (ij, jk) , kl -> il ").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'l']);
    }

    #[test]
    fn test_parse_error_no_arrow() {
        assert!(parse_einsum("ij,jk").is_err());
    }
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p strided-opteinsum parse`
Expected: All fail with `todo!()`

**Step 3: Implement the parser**

Recursive descent parser. Grammar:
```
einsum    := args_list '->' indices
args_list := arg (',' arg)*
arg       := '(' args_list ')' | indices
indices   := [a-z]*
```

```rust
use crate::error::{EinsumError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum EinsumNode {
    Leaf { ids: Vec<char>, tensor_index: usize },
    Contract { args: Vec<EinsumNode> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EinsumCode {
    pub root: EinsumNode,
    pub output_ids: Vec<char>,
}

pub fn parse_einsum(s: &str) -> Result<EinsumCode> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let arrow = s.find("->").ok_or_else(|| {
        EinsumError::ParseError("missing '->' separator".into())
    })?;
    let lhs = &s[..arrow];
    let rhs = &s[arrow + 2..];
    let output_ids: Vec<char> = rhs.chars().collect();

    let mut tensor_counter = 0usize;
    let (root, _) = parse_args_list(lhs, &mut tensor_counter)?;

    Ok(EinsumCode { root, output_ids })
}

/// Parse a comma-separated list of args. Returns a Contract node.
fn parse_args_list(s: &str, counter: &mut usize) -> Result<(EinsumNode, usize)> {
    let parts = split_top_level(s)?;
    let mut args = Vec::new();
    for part in &parts {
        let (node, _) = parse_arg(part, counter)?;
        args.push(node);
    }
    Ok((EinsumNode::Contract { args }, *counter))
}

/// Parse a single arg: either `(args_list)` or bare indices.
fn parse_arg(s: &str, counter: &mut usize) -> Result<(EinsumNode, usize)> {
    if s.starts_with('(') && s.ends_with(')') {
        // Nested: strip parens and parse inner args_list
        let inner = &s[1..s.len() - 1];
        parse_args_list(inner, counter)
    } else if s.contains('(') || s.contains(')') {
        Err(EinsumError::ParseError(format!("unbalanced parentheses in '{}'", s)))
    } else {
        // Leaf: bare indices
        let ids: Vec<char> = s.chars().collect();
        let idx = *counter;
        *counter += 1;
        Ok((EinsumNode::Leaf { ids, tensor_index: idx }, *counter))
    }
}

/// Split a string by commas at the top level (respecting parentheses).
fn split_top_level(s: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '(' => { depth += 1; current.push(c); }
            ')' => {
                if depth == 0 {
                    return Err(EinsumError::ParseError("unbalanced ')'".into()));
                }
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
            }
            _ => { current.push(c); }
        }
    }
    if depth != 0 {
        return Err(EinsumError::ParseError("unbalanced '('".into()));
    }
    if !current.is_empty() {
        parts.push(current);
    }
    Ok(parts)
}
```

**Step 4: Run tests**

Run: `cargo test -p strided-opteinsum parse`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add strided-opteinsum/src/parse.rs
git commit -m "feat(opteinsum): implement nested einsum string parser"
```

---

### Task 5: Implement single-tensor operations

**Files:**
- Modify: `strided-opteinsum/src/single_tensor.rs`

Handles: permutation, diagonal extraction, partial trace, full trace.
Uses `diagonal_view` from Task 1 and `reduce_axis` from strided-kernel.

**Step 1: Write failing tests**

```rust
use strided_view::{StridedArray, StridedView};
use crate::error::Result;

/// Execute a single-tensor einsum operation.
///
/// Given input with axis IDs `input_ids` and desired `output_ids`:
/// 1. Identify repeated indices → diagonal_view (stride trick)
/// 2. Identify indices to sum out → reduce_axis
/// 3. Permute to output order
pub fn single_tensor_einsum<T>(
    src: &StridedView<T>,
    input_ids: &[char],
    output_ids: &[char],
) -> Result<StridedArray<T>>
where
    T: Copy
        + strided_view::ElementOpApply
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + num_traits::Zero,
{
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_permutation_only() {
        // ijk -> kji
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 3, 4], |idx| {
            (idx[0] * 12 + idx[1] * 4 + idx[2]) as f64
        });
        let result = single_tensor_einsum(&arr.view(), &['i', 'j', 'k'], &['k', 'j', 'i']).unwrap();
        assert_eq!(result.dims(), &[4, 3, 2]);
        // result[k,j,i] = arr[i,j,k]
        assert_abs_diff_eq!(result.get(&[0, 0, 0]), 0.0);
        assert_abs_diff_eq!(result.get(&[3, 2, 1]), 23.0); // arr[1,2,3] = 12+8+3 = 23
    }

    #[test]
    fn test_full_trace() {
        // ii -> (scalar)
        let mut arr = StridedArray::<f64>::col_major(&[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                arr.set(&[i, j], (i * 10 + j) as f64);
            }
        }
        // trace = A[0,0] + A[1,1] + A[2,2] = 0 + 11 + 22 = 33
        let result = single_tensor_einsum(&arr.view(), &['i', 'i'], &[]).unwrap();
        assert_eq!(result.dims(), &[]);
        assert_abs_diff_eq!(result.data()[0], 33.0);
    }

    #[test]
    fn test_partial_trace() {
        // iij -> j
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 2, 3], |idx| {
            (idx[0] * 6 + idx[1] * 3 + idx[2]) as f64
        });
        // result[j] = sum_i A[i,i,j] = A[0,0,j] + A[1,1,j]
        // A[0,0,:] = [0,1,2], A[1,1,:] = [9,10,11]
        // result = [9, 11, 13]
        let result = single_tensor_einsum(&arr.view(), &['i', 'i', 'j'], &['j']).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_abs_diff_eq!(result.get(&[0]), 9.0);
        assert_abs_diff_eq!(result.get(&[1]), 11.0);
        assert_abs_diff_eq!(result.get(&[2]), 13.0);
    }

    #[test]
    fn test_diagonal_extraction() {
        // ijj -> ij
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 3, 3], |idx| {
            (idx[0] * 9 + idx[1] * 3 + idx[2]) as f64
        });
        // result[i,j] = A[i,j,j]
        let result = single_tensor_einsum(&arr.view(), &['i', 'j', 'j'], &['i', 'j']).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
        // A[0,0,0]=0, A[0,1,1]=4, A[0,2,2]=8
        // A[1,0,0]=9, A[1,1,1]=13, A[1,2,2]=17
        assert_abs_diff_eq!(result.get(&[0, 0]), 0.0);
        assert_abs_diff_eq!(result.get(&[0, 1]), 4.0);
        assert_abs_diff_eq!(result.get(&[0, 2]), 8.0);
        assert_abs_diff_eq!(result.get(&[1, 0]), 9.0);
        assert_abs_diff_eq!(result.get(&[1, 1]), 13.0);
        assert_abs_diff_eq!(result.get(&[1, 2]), 17.0);
    }

    #[test]
    fn test_sum_axis() {
        // ij -> i (sum over j)
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| {
            (idx[0] * 3 + idx[1]) as f64
        });
        // result[i] = sum_j A[i,j]
        // result[0] = 0+1+2 = 3, result[1] = 3+4+5 = 12
        let result = single_tensor_einsum(&arr.view(), &['i', 'j'], &['i']).unwrap();
        assert_eq!(result.dims(), &[1, 2]);  // reduce_axis may keep rank
        // Check values
    }
}
```

Note: The `test_sum_axis` test may need adjustment based on how `reduce_axis` shapes its output. Adjust assertions during implementation.

**Step 2: Run tests to verify failure**

Run: `cargo test -p strided-opteinsum single_tensor`
Expected: Compile error / `todo!()`

**Step 3: Implement**

```rust
use std::collections::HashMap;
use strided_kernel::{copy_into, reduce_axis};
use strided_view::{StridedArray, StridedView};
use crate::error::{EinsumError, Result};

pub fn single_tensor_einsum<T>(
    src: &StridedView<T>,
    input_ids: &[char],
    output_ids: &[char],
) -> Result<StridedArray<T>>
where
    T: Copy
        + strided_view::ElementOpApply
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + num_traits::Zero,
{
    // Step 1: Find repeated index pairs for diagonal_view
    let mut seen: HashMap<char, usize> = HashMap::new();
    let mut diag_pairs: Vec<(usize, usize)> = Vec::new();
    for (i, &id) in input_ids.iter().enumerate() {
        if let Some(&prev) = seen.get(&id) {
            diag_pairs.push((prev, i));
        } else {
            seen.insert(id, i);
        }
    }

    // Apply diagonal_view and compute the unique IDs after diagonal extraction
    let diag_view;
    let unique_ids: Vec<char>;
    if diag_pairs.is_empty() {
        diag_view = src.clone();
        unique_ids = input_ids.to_vec();
    } else {
        diag_view = src.diagonal_view(&diag_pairs)?;
        // After diagonal_view, the higher axis of each pair is removed.
        // Build unique_ids: keep only first occurrence of each ID.
        let mut removed = std::collections::HashSet::new();
        for &(_, hi) in &diag_pairs {
            removed.insert(hi);
        }
        unique_ids = input_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed.contains(i))
            .map(|(_, &id)| id)
            .collect();
    }

    // Step 2: Find axes to sum out (in unique_ids but not in output_ids)
    let output_set: std::collections::HashSet<char> = output_ids.iter().cloned().collect();
    let axes_to_reduce: Vec<usize> = unique_ids
        .iter()
        .enumerate()
        .filter(|(_, id)| !output_set.contains(id))
        .map(|(i, _)| i)
        .collect();

    // Reduce axes from back to front (preserves earlier indices)
    let mut current_view = diag_view;
    let mut current_ids = unique_ids;
    for &ax in axes_to_reduce.iter().rev() {
        let reduced = reduce_axis(
            &current_view,
            ax,
            |x| x,
            |a, b| a + b,
            T::zero(),
        )?;
        current_ids.remove(ax);
        current_view = reduced.view().clone();
        // We need to keep the owned array alive — store it
        // Actually, reduce_axis returns StridedArray, we need to handle ownership
    }

    // Hmm, ownership issue with reduce_axis returning StridedArray
    // but current_view borrows it. Let's restructure:
    // Reduce all axes, keeping owned arrays.

    // Re-do: reduce axes one by one, holding onto owned results.
    // Reset
    let mut current_arr: Option<StridedArray<T>> = None;
    let working_view = if diag_pairs.is_empty() {
        src.clone()
    } else {
        src.diagonal_view(&diag_pairs)?
    };

    let mut current_ids_2 = if diag_pairs.is_empty() {
        input_ids.to_vec()
    } else {
        let mut removed = std::collections::HashSet::new();
        for &(_, hi) in &diag_pairs {
            removed.insert(hi);
        }
        input_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed.contains(i))
            .map(|(_, &id)| id)
            .collect()
    };

    // Identify axes to reduce
    let axes_to_reduce_2: Vec<usize> = current_ids_2
        .iter()
        .enumerate()
        .filter(|(_, id)| !output_set.contains(id))
        .map(|(i, _)| i)
        .rev()
        .collect();

    if axes_to_reduce_2.is_empty() && diag_pairs.is_empty() {
        // Pure permutation
        let perm = compute_permutation(&current_ids_2, output_ids);
        let permuted = working_view.permute(&perm)?;
        let mut out = StridedArray::<T>::col_major(permuted.dims());
        copy_into(&mut out.view_mut(), &permuted)?;
        return Ok(out);
    }

    // First reduce uses the diagonal view, subsequent uses owned arrays
    let mut needs_first = true;
    for &ax in &axes_to_reduce_2 {
        let view = if needs_first {
            needs_first = false;
            working_view.clone()
        } else {
            current_arr.as_ref().unwrap().view()
        };
        let reduced = reduce_axis(&view, ax, |x| x, |a, b| a + b, T::zero())?;
        current_ids_2.remove(ax);
        current_arr = Some(reduced);
    }

    // If we only did diagonal (no reduction)
    if current_arr.is_none() {
        let perm = compute_permutation(&current_ids_2, output_ids);
        let permuted = working_view.permute(&perm)?;
        let mut out = StridedArray::<T>::col_major(permuted.dims());
        copy_into(&mut out.view_mut(), &permuted)?;
        return Ok(out);
    }

    // Step 3: Permute to output order
    let arr = current_arr.unwrap();
    if output_ids.is_empty() {
        // Scalar output
        return Ok(arr);
    }

    let perm = compute_permutation(&current_ids_2, output_ids);
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        // Already in order
        return Ok(arr);
    }
    let permuted = arr.view().permute(&perm)?;
    let mut out = StridedArray::<T>::col_major(permuted.dims());
    copy_into(&mut out.view_mut(), &permuted)?;
    Ok(out)
}

fn compute_permutation(current: &[char], target: &[char]) -> Vec<usize> {
    target
        .iter()
        .map(|t| current.iter().position(|c| c == t).unwrap())
        .collect()
}
```

Note: The above implementation has ownership complexity with `reduce_axis` returning owned arrays while we need views for subsequent reductions. The implementer should clean this up — the key pattern is: hold the owned `StridedArray` from each reduce step, then call `.view()` on it for the next step.

**Step 4: Run tests and iterate**

Run: `cargo test -p strided-opteinsum single_tensor`
Expected: Tests pass (may need adjustments to handle `reduce_axis` output shape)

**Step 5: Commit**

```bash
git add strided-opteinsum/src/single_tensor.rs
git commit -m "feat(opteinsum): implement single-tensor einsum (permute, trace, diagonal)"
```

---

### Task 6: Implement EinsumCode evaluation (binary contraction)

**Files:**
- Modify: `strided-opteinsum/src/expr.rs`

**Step 1: Write failing tests**

```rust
use crate::error::Result;
use crate::operand::EinsumOperand;
use crate::parse::{parse_einsum, EinsumCode};
use strided_view::StridedArray;

impl EinsumCode {
    /// Evaluate the einsum with provided operands.
    /// Operands are matched to leaves by tensor_index order.
    pub fn evaluate<'a>(&self, operands: Vec<EinsumOperand<'a>>) -> Result<EinsumOperand<'static>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_f64_array(dims: &[usize], data: Vec<f64>) -> StridedArray<f64> {
        StridedArray::from_parts(data, dims, &strided_view::row_major_strides(dims), 0).unwrap()
    }

    #[test]
    fn test_matmul() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        let a = make_f64_array(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64_array(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let result = code.evaluate(vec![a.into(), b.into()]).unwrap();
        // C = A*B = [[19,22],[43,50]]
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[2, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 22.0);
                assert_abs_diff_eq!(arr.get(&[1, 0]), 43.0);
                assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_nested_three_tensor() {
        // (ij,jk),kl->il : first ij,jk->ik, then ik,kl->il
        let code = parse_einsum("(ij,jk),kl->il").unwrap();
        let a = make_f64_array(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f64_array(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let c = make_f64_array(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
        let result = code.evaluate(vec![a.into(), b.into(), c.into()]).unwrap();
        // AB = [[4,2],[10,5]], AB*I = [[4,2],[10,5]]
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[2, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 4.0);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 2.0);
                assert_abs_diff_eq!(arr.get(&[1, 0]), 10.0);
                assert_abs_diff_eq!(arr.get(&[1, 1]), 5.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_outer_product() {
        let code = parse_einsum("i,j->ij").unwrap();
        let a = make_f64_array(&[3], vec![1.0, 2.0, 3.0]);
        let b = make_f64_array(&[2], vec![10.0, 20.0]);
        let result = code.evaluate(vec![a.into(), b.into()]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[3, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 10.0);
                assert_abs_diff_eq!(arr.get(&[2, 1]), 60.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_single_tensor_permute() {
        let code = parse_einsum("ij->ji").unwrap();
        let a = make_f64_array(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = code.evaluate(vec![a.into()]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[3, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 1.0);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 4.0);
                assert_abs_diff_eq!(arr.get(&[2, 0]), 3.0);
                assert_abs_diff_eq!(arr.get(&[2, 1]), 6.0);
            }
            _ => panic!("expected F64"),
        }
    }
}
```

**Step 2: Run to verify failure**

Run: `cargo test -p strided-opteinsum expr`
Expected: `todo!()`

**Step 3: Implement evaluation**

The evaluation logic in `expr.rs`:
1. Collect all leaves from the tree (by tensor_index order)
2. Match with provided operands
3. Recursively evaluate the tree:
   - Leaf → return operand
   - Contract (1 child) → single_tensor_einsum
   - Contract (2 children) → call einsum2_into
   - Contract (3+ children) → omeco fallback (Task 7)

For pairwise contraction, both operands must have the same scalar type.
If mixed, promote f64 to c64.

The intermediate output_ids for each Contract node must be computed (like OMEinsum.jl's `filliys!`):
- output_ids = (indices in final output) ∪ (indices shared with sibling nodes)

Reference: `mdarray-einsum/src/lib.rs` function `compute_intermediate_output`.

```rust
use std::collections::{HashMap, HashSet};
use crate::error::{EinsumError, Result};
use crate::operand::{EinsumOperand, StridedData};
use crate::parse::{EinsumCode, EinsumNode};
use crate::single_tensor::single_tensor_einsum;
use strided_view::{StridedArray, StridedView};
use strided_einsum2::einsum2_into;
use num_traits::{One, Zero};

impl EinsumCode {
    pub fn evaluate<'a>(&self, operands: Vec<EinsumOperand<'a>>) -> Result<EinsumOperand<'static>> {
        let final_output = &self.output_ids;
        evaluate_node(&self.root, &operands, final_output)
    }
}

fn evaluate_node<'a>(
    node: &EinsumNode,
    operands: &[EinsumOperand<'a>],
    final_output: &[char],
) -> Result<EinsumOperand<'static>> {
    match node {
        EinsumNode::Leaf { ids: _, tensor_index } => {
            // Clone operand into owned
            let op = &operands[*tensor_index];
            // Convert to owned
            Ok(to_owned(op))
        }
        EinsumNode::Contract { args } => {
            if args.len() == 1 {
                // Single-tensor operation
                let child = evaluate_node(&args[0], operands, final_output)?;
                let child_ids = collect_ids(&args[0], operands);
                return eval_single(child, &child_ids, final_output);
            }

            // Compute intermediate output_ids for this node
            let node_output_ids = compute_node_output_ids(node, final_output);

            if args.len() == 2 {
                let left = evaluate_node(&args[0], operands, final_output)?;
                let right = evaluate_node(&args[1], operands, final_output)?;
                let left_ids = collect_node_ids(&args[0], operands);
                let right_ids = collect_node_ids(&args[1], operands);
                return eval_pair(left, &left_ids, right, &right_ids, &node_output_ids);
            }

            // 3+ children: use omeco fallback or left-to-right
            eval_multi(args, operands, final_output, &node_output_ids)
        }
    }
}

/// Pairwise contraction via einsum2_into.
fn eval_pair(
    left: EinsumOperand<'static>,
    left_ids: &[char],
    right: EinsumOperand<'static>,
    right_ids: &[char],
    output_ids: &[char],
) -> Result<EinsumOperand<'static>> {
    // Determine output type and promote if needed
    match (left.is_f64(), right.is_f64()) {
        (true, true) => {
            // Both f64
            eval_pair_typed::<f64>(left, left_ids, right, right_ids, output_ids)
        }
        (false, false) => {
            // Both c64
            eval_pair_typed::<num_complex::Complex64>(left, left_ids, right, right_ids, output_ids)
        }
        _ => {
            // Mixed: promote both to c64
            let left_c = left.to_c64_owned();
            let right_c = right.to_c64_owned();
            eval_pair_typed::<num_complex::Complex64>(left_c, left_ids, right_c, right_ids, output_ids)
        }
    }
}

fn eval_pair_typed<T>(
    left: EinsumOperand<'static>,
    left_ids: &[char],
    right: EinsumOperand<'static>,
    right_ids: &[char],
    output_ids: &[char],
) -> Result<EinsumOperand<'static>>
where
    T: strided_einsum2::Scalar + Default + strided_view::ElementOpApply,
{
    // Extract views
    // ... (extract StridedView<T> from the operands)
    // Allocate output
    // Call einsum2_into
    // Return Owned result
    todo!("extract views, allocate output, call einsum2_into")
}

/// Collect all index IDs that appear in a node's subtree.
fn collect_node_ids(node: &EinsumNode, operands: &[EinsumOperand]) -> Vec<char> {
    match node {
        EinsumNode::Leaf { ids, .. } => ids.clone(),
        EinsumNode::Contract { args } => {
            // Union of all children's IDs, preserving first-seen order
            let mut seen = HashSet::new();
            let mut result = Vec::new();
            for arg in args {
                for id in collect_node_ids(arg, operands) {
                    if seen.insert(id) {
                        result.push(id);
                    }
                }
            }
            result
        }
    }
}

/// Compute output IDs for a contraction node.
/// Keep an ID if it appears in final_output OR in multiple children.
fn compute_node_output_ids(node: &EinsumNode, final_output: &[char]) -> Vec<char> {
    let final_set: HashSet<char> = final_output.iter().cloned().collect();
    match node {
        EinsumNode::Leaf { ids, .. } => ids.clone(),
        EinsumNode::Contract { args } => {
            // Count how many children each ID appears in
            let mut id_child_count: HashMap<char, usize> = HashMap::new();
            for arg in args {
                let child_ids: HashSet<char> = collect_node_ids_set(arg);
                for id in child_ids {
                    *id_child_count.entry(id).or_insert(0) += 1;
                }
            }
            // Keep: in final_output OR appears in 2+ children (shared/batch)
            let mut result = Vec::new();
            let mut seen = HashSet::new();
            for arg in args {
                for id in collect_node_ids_flat(arg) {
                    if seen.insert(id) {
                        if final_set.contains(&id) || id_child_count.get(&id).copied().unwrap_or(0) >= 2 {
                            result.push(id);
                        }
                    }
                }
            }
            result
        }
    }
}
```

Note: The full implementation of `eval_pair_typed` requires extracting `StridedView<T>` from `EinsumOperand` (matching on `F64`/`C64` variants) and calling `einsum2_into`. This involves some boilerplate for type dispatch. The implementer should write helper methods on `EinsumOperand` and `StridedData` to extract typed views.

**Step 4: Run tests and iterate**

Run: `cargo test -p strided-opteinsum expr`
Expected: Tests pass

**Step 5: Commit**

```bash
git add strided-opteinsum/src/expr.rs
git commit -m "feat(opteinsum): implement EinsumCode evaluation with binary contraction"
```

---

### Task 7: Implement omeco fallback for 3+ tensors

**Files:**
- Modify: `strided-opteinsum/src/expr.rs`

When a Contract node has 3+ children and the user hasn't nested them further, fall back to omeco to find optimal contraction order.

**Step 1: Write failing test**

```rust
#[test]
fn test_three_tensor_flat() {
    // ij,jk,kl->il — flat, will use omeco
    let code = parse_einsum("ij,jk,kl->il").unwrap();
    let a = make_f64_array(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = make_f64_array(&[3, 4], vec![1.0; 12]);
    let c = make_f64_array(&[4, 2], vec![1.0; 8]);
    let result = code.evaluate(vec![a.into(), b.into(), c.into()]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 2]);
        }
        _ => panic!("expected F64"),
    }
}
```

**Step 2: Implement `eval_multi`**

```rust
use omeco::{EinCode, GreedyMethod, CodeOptimizer};

fn eval_multi(
    args: &[EinsumNode],
    operands: &[EinsumOperand],
    final_output: &[char],
    node_output_ids: &[char],
) -> Result<EinsumOperand<'static>> {
    // Collect each child's IDs
    let child_ids: Vec<Vec<char>> = args.iter()
        .map(|a| collect_node_ids_flat(a))
        .collect();

    // Build dimension sizes from operands
    let mut dim_sizes: HashMap<char, usize> = HashMap::new();
    for (arg, ids) in args.iter().zip(&child_ids) {
        // Evaluate to get dims, or inspect leaves
        collect_dim_sizes(arg, operands, &mut dim_sizes);
    }

    // Use omeco to optimize
    let ixs: Vec<Vec<char>> = child_ids.clone();
    let code = EinCode::new(ixs.clone(), node_output_ids.to_vec());
    let optimizer = GreedyMethod::default();
    let nested = optimizer.optimize(&code, &dim_sizes);

    // Execute the nested contraction tree
    // Evaluate children first, then follow omeco's tree
    let mut evaluated: Vec<(Vec<char>, EinsumOperand<'static>)> = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        let child_result = evaluate_node(arg, operands, final_output)?;
        evaluated.push((child_ids[i].clone(), child_result));
    }

    // Execute omeco's path: convert NestedEinsum to contraction steps
    execute_omeco_path(&nested, &mut evaluated, node_output_ids)
}
```

Reference: Port `mdarray-einsum/src/optimizer.rs` functions `nested_to_steps` and `steps_to_path`, adapted to work with `EinsumOperand` instead of `Tensor`.

**Step 3: Run tests**

Run: `cargo test -p strided-opteinsum test_three_tensor_flat`
Expected: PASS

**Step 4: Commit**

```bash
git add strided-opteinsum/src/expr.rs
git commit -m "feat(opteinsum): add omeco fallback for 3+ tensor contraction"
```

---

### Task 8: Implement TypedTensor

**Files:**
- Modify: `strided-opteinsum/src/typed_tensor.rs`

Port from `mdarray-einsum/src/typed_tensor.rs`, adapted to use `StridedArray` instead of `Tensor`.

**Step 1: Write failing tests**

```rust
use num_complex::Complex64;
use strided_view::StridedArray;

pub enum TypedTensor {
    F64(StridedArray<f64>),
    C64(StridedArray<Complex64>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_typed_f64() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let t = TypedTensor::F64(arr);
        assert!(t.is_f64());
        assert!(!t.is_c64());
        assert_eq!(t.dims(), &[2, 3]);
    }

    #[test]
    fn test_typed_c64() {
        let arr = StridedArray::<Complex64>::col_major(&[3, 4]);
        let t = TypedTensor::C64(arr);
        assert!(t.is_c64());
    }

    #[test]
    fn test_promote_to_c64() {
        let mut arr = StridedArray::<f64>::col_major(&[2]);
        arr.data_mut()[0] = 3.0;
        arr.data_mut()[1] = 7.0;
        let t = TypedTensor::F64(arr);
        let promoted = t.to_c64();
        match promoted {
            TypedTensor::C64(a) => {
                assert_abs_diff_eq!(a.data()[0].re, 3.0);
                assert_abs_diff_eq!(a.data()[1].re, 7.0);
                assert_abs_diff_eq!(a.data()[0].im, 0.0);
            }
            _ => panic!("expected C64"),
        }
    }

    #[test]
    fn test_demote_to_f64() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2]);
        arr.data_mut()[0] = Complex64::new(1.0, 0.0);
        arr.data_mut()[1] = Complex64::new(2.0, 1e-16);
        let t = TypedTensor::try_demote_to_f64(arr);
        assert!(t.is_f64()); // imaginary parts < threshold
    }

    #[test]
    fn test_no_demote_if_complex() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2]);
        arr.data_mut()[0] = Complex64::new(1.0, 0.5);
        arr.data_mut()[1] = Complex64::new(2.0, 0.0);
        let t = TypedTensor::try_demote_to_f64(arr);
        assert!(t.is_c64()); // has significant imaginary part
    }

    #[test]
    fn test_needs_c64_promotion() {
        let f = TypedTensor::F64(StridedArray::<f64>::col_major(&[1]));
        let c = TypedTensor::C64(StridedArray::<Complex64>::col_major(&[1]));
        assert!(!needs_c64_promotion(&[&f]));
        assert!(needs_c64_promotion(&[&f, &c]));
        assert!(needs_c64_promotion(&[&c]));
    }
}
```

**Step 2: Implement**

```rust
use num_complex::Complex64;
use strided_view::StridedArray;

pub enum TypedTensor {
    F64(StridedArray<f64>),
    C64(StridedArray<Complex64>),
}

const DEMOTE_THRESHOLD: f64 = 1e-15;

impl TypedTensor {
    pub fn is_f64(&self) -> bool { matches!(self, TypedTensor::F64(_)) }
    pub fn is_c64(&self) -> bool { matches!(self, TypedTensor::C64(_)) }

    pub fn dims(&self) -> &[usize] {
        match self {
            TypedTensor::F64(a) => a.dims(),
            TypedTensor::C64(a) => a.dims(),
        }
    }

    pub fn to_c64(self) -> TypedTensor {
        match self {
            TypedTensor::C64(_) => self,
            TypedTensor::F64(a) => {
                let c64_data: Vec<Complex64> = a.data().iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                let arr = StridedArray::from_parts(c64_data, a.dims(), a.strides(), 0).unwrap();
                TypedTensor::C64(arr)
            }
        }
    }

    pub fn try_demote_to_f64(arr: StridedArray<Complex64>) -> TypedTensor {
        let all_real = arr.data().iter().all(|c| c.im.abs() < DEMOTE_THRESHOLD);
        if all_real {
            let f64_data: Vec<f64> = arr.data().iter().map(|c| c.re).collect();
            let f_arr = StridedArray::from_parts(f64_data, arr.dims(), arr.strides(), 0).unwrap();
            TypedTensor::F64(f_arr)
        } else {
            TypedTensor::C64(arr)
        }
    }
}

pub fn needs_c64_promotion(inputs: &[&TypedTensor]) -> bool {
    inputs.iter().any(|t| t.is_c64())
}
```

**Step 3: Run tests**

Run: `cargo test -p strided-opteinsum typed_tensor`
Expected: All PASS

**Step 4: Commit**

```bash
git add strided-opteinsum/src/typed_tensor.rs
git commit -m "feat(opteinsum): implement TypedTensor with F64/C64 promotion and demotion"
```

---

### Task 9: Wire up lib.rs public API

**Files:**
- Modify: `strided-opteinsum/src/lib.rs`

**Step 1: Add convenience function**

```rust
pub mod error;
pub mod operand;
pub mod parse;
pub mod expr;
pub mod single_tensor;
pub mod typed_tensor;

pub use error::{EinsumError, Result};
pub use operand::{EinsumOperand, StridedData};
pub use parse::{parse_einsum, EinsumCode, EinsumNode};
pub use typed_tensor::{needs_c64_promotion, TypedTensor};

/// Convenience: parse + evaluate in one call.
///
/// ```rust
/// let result = einsum("(ij,jk),kl->il", vec![a.into(), b.into(), c.into()])?;
/// ```
pub fn einsum<'a>(
    notation: &str,
    operands: Vec<EinsumOperand<'a>>,
) -> Result<EinsumOperand<'static>> {
    let code = parse_einsum(notation)?;
    code.evaluate(operands)
}
```

**Step 2: Add integration test**

Create `strided-opteinsum/tests/integration.rs`:

```rust
use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use strided_opteinsum::{einsum, EinsumOperand};
use strided_view::StridedArray;

fn make_f64(dims: &[usize], data: Vec<f64>) -> EinsumOperand<'static> {
    let strides = strided_view::row_major_strides(dims);
    StridedArray::from_parts(data, dims, &strides, 0).unwrap().into()
}

#[test]
fn test_matmul_e2e() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_nested_chain_e2e() {
    let a = make_f64(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = make_f64(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let result = einsum("(ij,jk),kl->il", vec![a, b, c]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 2]);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_trace_e2e() {
    let a = make_f64(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let result = einsum("ii->", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            assert_abs_diff_eq!(data.as_array().data()[0], 6.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_mixed_f64_c64() {
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let b_data = vec![
        Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 1.0), Complex64::new(3.0, 0.0),
    ];
    let b_strides = strided_view::row_major_strides(&[2, 2]);
    let b_arr = StridedArray::from_parts(b_data, &[2, 2], &b_strides, 0).unwrap();
    let b = EinsumOperand::from(b_arr);
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    assert!(result.is_c64());
}
```

**Step 3: Run all tests**

Run: `cargo test -p strided-opteinsum`
Expected: All PASS

**Step 4: Commit**

```bash
git add strided-opteinsum/src/lib.rs strided-opteinsum/tests/integration.rs
git commit -m "feat(opteinsum): wire up public API with einsum() convenience function"
```

---

### Task 10: Migrate strided-einsum2 trace.rs to use kernel's reduce_axis

**Files:**
- Modify: `strided-einsum2/src/trace.rs`
- Modify: `strided-einsum2/src/lib.rs`

This task replaces einsum2's custom trace reduction with calls to strided-kernel's
`reduce_axis`, making the code simpler and DRY.

**Step 1: Rewrite trace.rs**

Replace the custom MultiIndex-based loop with sequential `reduce_axis` calls:

```rust
use strided_kernel::reduce_axis;
use strided_view::{ElementOp, ElementOpApply, StridedArray, StridedView};

/// Reduce trace axes (axes appearing only in this operand, not in output).
/// Sums over each trace axis sequentially.
pub fn reduce_trace_axes<T, Op>(
    src: &StridedView<T, Op>,
    trace_axes: &[usize],
) -> strided_view::Result<StridedArray<T>>
where
    T: Copy + ElementOpApply + Send + Sync + std::ops::Add<Output = T> + num_traits::Zero,
    Op: ElementOp,
{
    assert!(!trace_axes.is_empty(), "trace_axes must not be empty");

    // Sort in descending order so we can reduce from back to front
    let mut axes: Vec<usize> = trace_axes.to_vec();
    axes.sort_unstable();
    axes.reverse();

    let first_reduced = reduce_axis(src, axes[0], |x| x, |a, b| a + b, T::zero())?;

    let mut current = first_reduced;
    for &ax in &axes[1..] {
        current = reduce_axis(&current.view(), ax, |x| x, |a, b| a + b, T::zero())?;
    }

    Ok(current)
}
```

**Step 2: Run existing einsum2 tests**

Run: `cargo test -p strided-einsum2`
Expected: All existing tests still PASS

**Step 3: Commit**

```bash
git add strided-einsum2/src/trace.rs
git commit -m "refactor(einsum2): simplify trace.rs using kernel's reduce_axis"
```

---

### Task 11: Final checks

**Step 1: Format check**

Run: `cargo fmt --check`
If fails: `cargo fmt`

**Step 2: Full test suite**

Run: `cargo test`
Expected: All tests across all crates PASS

**Step 3: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: apply cargo fmt"
```
