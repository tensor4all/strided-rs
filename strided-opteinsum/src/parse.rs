/// Parsed contraction tree node (no operands, just structure).
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
pub fn parse_einsum(s: &str) -> crate::Result<EinsumCode> {
    // Strip all whitespace
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();

    // Split on "->"
    let arrow_pos = s
        .find("->")
        .ok_or_else(|| crate::EinsumError::ParseError("missing '->' in einsum string".into()))?;
    let lhs = &s[..arrow_pos];
    let rhs = &s[arrow_pos + 2..];

    // Parse output indices
    let output_ids: Vec<char> = rhs.chars().collect();
    for &c in &output_ids {
        if !c.is_alphabetic() {
            return Err(crate::EinsumError::ParseError(format!(
                "invalid character '{}' in output indices",
                c
            )));
        }
    }

    // Parse LHS as args_list.
    // Empty LHS (e.g. "->ii") is a single scalar operand with no indices.
    // Leading/trailing commas (e.g. ",k->k") produce scalar operands too.
    let mut counter: usize = 0;
    let root = parse_args_list(lhs, &mut counter)?;

    Ok(EinsumCode { root, output_ids })
}

/// Parse a comma-separated args list at the current level, returning a `Contract` node.
///
/// An empty string produces a single scalar Leaf (0-index operand).
fn parse_args_list(s: &str, counter: &mut usize) -> crate::Result<EinsumNode> {
    let parts = split_top_level(s)?;
    if parts.is_empty() {
        // Empty string (no commas, no chars) → single scalar operand
        return parse_arg("", counter).map(|leaf| EinsumNode::Contract { args: vec![leaf] });
    }
    let mut args = Vec::with_capacity(parts.len());
    for part in parts {
        args.push(parse_arg(&part, counter)?);
    }
    // If there is exactly one arg and it is a Contract, unwrap it to avoid
    // redundant nesting from outer parentheses like "((ij,jk),(kl,lm))".
    if args.len() == 1 {
        if let EinsumNode::Contract { .. } = &args[0] {
            return Ok(args.into_iter().next().unwrap());
        }
    }
    Ok(EinsumNode::Contract { args })
}

/// Parse a single arg: if wrapped in `(...)`, recursively parse inner as args_list;
/// otherwise it's a leaf with index characters.
fn parse_arg(s: &str, counter: &mut usize) -> crate::Result<EinsumNode> {
    if s.starts_with('(') && s.ends_with(')') {
        // Strip outer parens and recursively parse
        let inner = &s[1..s.len() - 1];
        parse_args_list(inner, counter)
    } else {
        // Empty string is a valid scalar operand (0-index tensor).
        // Leaf: validate all chars are alphabetic (ASCII or Unicode letters)
        for c in s.chars() {
            if !c.is_alphabetic() {
                return Err(crate::EinsumError::ParseError(format!(
                    "invalid character '{}' in index labels",
                    c
                )));
            }
        }
        let ids: Vec<char> = s.chars().collect();
        let tensor_index = *counter;
        *counter += 1;
        Ok(EinsumNode::Leaf { ids, tensor_index })
    }
}

/// Split a string by commas, respecting parenthesis nesting depth.
fn split_top_level(s: &str) -> crate::Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                if depth == 0 {
                    return Err(crate::EinsumError::ParseError("unbalanced ')'".into()));
                }
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                // Empty `current` is valid: it represents a scalar (0-index) operand.
                parts.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(c);
            }
        }
    }
    if depth != 0 {
        return Err(crate::EinsumError::ParseError("unbalanced '('".into()));
    }
    if !current.is_empty() || !parts.is_empty() {
        // Push final segment. Empty `current` after a comma is a valid scalar operand.
        parts.push(current);
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_flat() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'k']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec!['i', 'j'],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec!['j', 'k'],
                        tensor_index: 1
                    }
                );
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
                match &args[0] {
                    EinsumNode::Contract { args: inner } => {
                        assert_eq!(inner.len(), 2);
                        assert_eq!(
                            inner[0],
                            EinsumNode::Leaf {
                                ids: vec!['i', 'j'],
                                tensor_index: 0
                            }
                        );
                        assert_eq!(
                            inner[1],
                            EinsumNode::Leaf {
                                ids: vec!['j', 'k'],
                                tensor_index: 1
                            }
                        );
                    }
                    _ => panic!("expected inner Contract"),
                }
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec!['k', 'l'],
                        tensor_index: 2
                    }
                );
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
                match &args[0] {
                    EinsumNode::Contract { args: left } => {
                        assert_eq!(left.len(), 2);
                        assert_eq!(
                            left[0],
                            EinsumNode::Leaf {
                                ids: vec!['i', 'j'],
                                tensor_index: 0
                            }
                        );
                        assert_eq!(
                            left[1],
                            EinsumNode::Leaf {
                                ids: vec!['j', 'k'],
                                tensor_index: 1
                            }
                        );
                    }
                    _ => panic!("expected left Contract"),
                }
                match &args[1] {
                    EinsumNode::Contract { args: right } => {
                        assert_eq!(right.len(), 2);
                        assert_eq!(
                            right[0],
                            EinsumNode::Leaf {
                                ids: vec!['k', 'l'],
                                tensor_index: 2
                            }
                        );
                        assert_eq!(
                            right[1],
                            EinsumNode::Leaf {
                                ids: vec!['l', 'm'],
                                tensor_index: 3
                            }
                        );
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
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec!['i', 'j', 'k'],
                        tensor_index: 0
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_three_flat() {
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

    #[test]
    fn test_parse_scalar_operand_leading_comma() {
        // ",k->k" = scalar (tensor 0) + vector k (tensor 1)
        let code = parse_einsum(",k->k").unwrap();
        assert_eq!(code.output_ids, vec!['k']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec![],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec!['k'],
                        tensor_index: 1
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_scalar_operand_trailing_comma() {
        // "i,->i" = vector i (tensor 0) + scalar (tensor 1)
        let code = parse_einsum("i,->i").unwrap();
        assert_eq!(code.output_ids, vec!['i']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec!['i'],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec![],
                        tensor_index: 1
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_two_scalars() {
        // ",->": two scalar operands
        let code = parse_einsum(",->").unwrap();
        assert_eq!(code.output_ids, vec![]);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec![],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec![],
                        tensor_index: 1
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_scalar_between_tensors() {
        // "ij,,jk->ik" = tensor 0 (ij) + scalar (tensor 1) + tensor 2 (jk)
        let code = parse_einsum("ij,,jk->ik").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'k']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 3);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec!['i', 'j'],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec![],
                        tensor_index: 1
                    }
                );
                assert_eq!(
                    args[2],
                    EinsumNode::Leaf {
                        ids: vec!['j', 'k'],
                        tensor_index: 2
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_empty_lhs_scalar() {
        // "->ii" = single scalar operand with generative output
        let code = parse_einsum("->ii").unwrap();
        assert_eq!(code.output_ids, vec!['i', 'i']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 1);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec![],
                        tensor_index: 0
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_unicode_greek() {
        let code = parse_einsum("αβ,βγ->αγ").unwrap();
        assert_eq!(code.output_ids, vec!['α', 'γ']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec!['α', 'β'],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec!['β', 'γ'],
                        tensor_index: 1
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_unicode_mixed() {
        let code = parse_einsum("αi,iβ->αβ").unwrap();
        assert_eq!(code.output_ids, vec!['α', 'β']);
        match &code.root {
            EinsumNode::Contract { args } => {
                assert_eq!(args.len(), 2);
                assert_eq!(
                    args[0],
                    EinsumNode::Leaf {
                        ids: vec!['α', 'i'],
                        tensor_index: 0
                    }
                );
                assert_eq!(
                    args[1],
                    EinsumNode::Leaf {
                        ids: vec!['i', 'β'],
                        tensor_index: 1
                    }
                );
            }
            _ => panic!("expected Contract"),
        }
    }

    #[test]
    fn test_parse_unicode_nested() {
        let code = parse_einsum("(αβ,βγ),γδ->αδ").unwrap();
        assert_eq!(code.output_ids, vec!['α', 'δ']);
    }
}
