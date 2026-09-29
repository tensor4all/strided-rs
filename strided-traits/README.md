# strided-traits

Shared traits for the strided-rs ecosystem.

This crate contains scalar bounds and lazy element-operation traits used by
`strided-view`, `strided-kernel`, and downstream extension
crates. Most users depend on `strided-view` and `strided-kernel` instead.

Depend on `strided-traits` directly when implementing custom scalar types or
element operations for the lower-level crates.
