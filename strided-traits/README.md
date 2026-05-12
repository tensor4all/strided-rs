# strided-traits

Shared traits for the strided-rs ecosystem.

This crate contains scalar bounds and lazy element-operation traits used by
`strided-view`, `strided-kernel`, `strided-einsum2`, and downstream extension
crates. Most users should depend on the `strided-rs` facade crate instead.

Depend on `strided-traits` directly when implementing custom scalar types or
element operations for the lower-level crates.
