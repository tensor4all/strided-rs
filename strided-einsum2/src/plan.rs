//! Einsum2 plan: axis classification and permutation computation.

use crate::util::invert_perm;
use crate::AxisId;
use crate::EinsumError;
use std::collections::{HashMap, HashSet};

/// Pre-computed execution plan for a binary einsum contraction.
///
/// Classifies axes into groups and precomputes the permutations needed
/// to arrange operands for batched matrix multiplication.
#[derive(Debug, Clone)]
pub struct Einsum2Plan<ID: AxisId> {
    /// Batch axes: present in A, B, and C.
    pub batch: Vec<ID>,
    /// Left-output axes: present in A and C, not in B.
    pub lo: Vec<ID>,
    /// Right-output axes: present in B and C, not in A.
    pub ro: Vec<ID>,
    /// Contraction axes: present in A and B, not in C.
    pub sum: Vec<ID>,
    /// Left trace axes: present only in A.
    pub left_trace: Vec<ID>,
    /// Right trace axes: present only in B.
    pub right_trace: Vec<ID>,

    /// Permutation to reorder A to [batch, lo, sum] after trace reduction.
    pub left_perm: Vec<usize>,
    /// Permutation to reorder B to [batch, sum, ro] after trace reduction.
    pub right_perm: Vec<usize>,
    /// Permutation to reorder C from IC order to [batch, lo, ro] order.
    pub c_to_internal_perm: Vec<usize>,
}

impl<ID: AxisId> Einsum2Plan<ID> {
    /// Build a plan from axis labels.
    ///
    /// `ia`, `ib`, `ic` are the axis labels for A, B, C respectively.
    pub fn new(ia: &[ID], ib: &[ID], ic: &[ID]) -> Result<Self, EinsumError> {
        let ia_set: HashSet<&ID> = ia.iter().collect();
        let ib_set: HashSet<&ID> = ib.iter().collect();
        let ic_set: HashSet<&ID> = ic.iter().collect();

        // Validate: no duplicate axes within a single operand
        if ia_set.len() != ia.len() {
            return Err(EinsumError::DuplicateAxis(
                "left operand has duplicate axis labels".into(),
            ));
        }
        if ib_set.len() != ib.len() {
            return Err(EinsumError::DuplicateAxis(
                "right operand has duplicate axis labels".into(),
            ));
        }
        if ic_set.len() != ic.len() {
            return Err(EinsumError::DuplicateAxis(
                "output has duplicate axis labels".into(),
            ));
        }

        // Validate: every output axis must appear in at least one input
        for id in ic {
            if !ia_set.contains(id) && !ib_set.contains(id) {
                return Err(EinsumError::OrphanOutputAxis(format!("{:?}", id)));
            }
        }

        let mut batch = Vec::new();
        let mut lo = Vec::new();
        let mut sum = Vec::new();
        let mut left_trace = Vec::new();

        for id in ia {
            if ib_set.contains(id) {
                if ic_set.contains(id) {
                    batch.push(id.clone());
                } else {
                    sum.push(id.clone());
                }
            } else if ic_set.contains(id) {
                lo.push(id.clone());
            } else {
                left_trace.push(id.clone());
            }
        }

        let mut ro = Vec::new();
        let mut right_trace = Vec::new();

        for id in ib {
            if !ia_set.contains(id) {
                if ic_set.contains(id) {
                    ro.push(id.clone());
                } else {
                    right_trace.push(id.clone());
                }
            }
        }

        // Build left_perm: maps positions in ia (after trace removal) to [batch, lo, sum] order
        let ia_after_trace: Vec<&ID> = ia.iter().filter(|id| !left_trace.contains(id)).collect();
        let ia_pos: HashMap<&ID, usize> = ia_after_trace
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let left_perm: Vec<usize> = batch
            .iter()
            .chain(lo.iter())
            .chain(sum.iter())
            .map(|id| *ia_pos.get(id).expect("left_perm: axis not found"))
            .collect();

        // Build right_perm: maps positions in ib (after trace removal) to [batch, sum, ro] order
        let ib_after_trace: Vec<&ID> = ib.iter().filter(|id| !right_trace.contains(id)).collect();
        let ib_pos: HashMap<&ID, usize> = ib_after_trace
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let right_perm: Vec<usize> = batch
            .iter()
            .chain(sum.iter())
            .chain(ro.iter())
            .map(|id| *ib_pos.get(id).expect("right_perm: axis not found"))
            .collect();

        // Build c_to_internal_perm: maps IC order to [batch, lo, ro] order
        let ic_pos: HashMap<&ID, usize> = ic.iter().enumerate().map(|(i, id)| (id, i)).collect();
        let c_to_internal_perm: Vec<usize> = batch
            .iter()
            .chain(lo.iter())
            .chain(ro.iter())
            .map(|id| *ic_pos.get(id).expect("c_to_internal_perm: axis not found"))
            .collect();

        Ok(Einsum2Plan {
            batch,
            lo,
            ro,
            sum,
            left_trace,
            right_trace,
            left_perm,
            right_perm,
            c_to_internal_perm,
        })
    }

    /// Get the indices of left_trace axes in the original `ia` array.
    pub fn left_trace_indices(&self, ia: &[ID]) -> Vec<usize> {
        self.left_trace
            .iter()
            .filter_map(|id| ia.iter().position(|x| x == id))
            .collect()
    }

    /// Get the indices of right_trace axes in the original `ib` array.
    pub fn right_trace_indices(&self, ib: &[ID]) -> Vec<usize> {
        self.right_trace
            .iter()
            .filter_map(|id| ib.iter().position(|x| x == id))
            .collect()
    }

    /// Get the inverse of c_to_internal_perm (maps `\[batch, lo, ro\]` back to IC order).
    pub fn internal_to_c_perm(&self) -> Vec<usize> {
        invert_perm(&self.c_to_internal_perm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_matmul() {
        // ij,jk->ik
        let plan = Einsum2Plan::new(&[0u32, 1], &[1u32, 2], &[0u32, 2]).unwrap();
        assert_eq!(plan.batch, vec![] as Vec<u32>);
        assert_eq!(plan.lo, vec![0]);
        assert_eq!(plan.ro, vec![2]);
        assert_eq!(plan.sum, vec![1]);
        assert!(plan.left_trace.is_empty());
        assert!(plan.right_trace.is_empty());
    }

    #[test]
    fn test_classify_batched_matmul() {
        // bij,bjk->bik
        let plan = Einsum2Plan::new(&[0u32, 1, 2], &[0u32, 2, 3], &[0u32, 1, 3]).unwrap();
        assert_eq!(plan.batch, vec![0]);
        assert_eq!(plan.lo, vec![1]);
        assert_eq!(plan.ro, vec![3]);
        assert_eq!(plan.sum, vec![2]);
    }

    #[test]
    fn test_classify_outer_product() {
        // i,j->ij
        let plan = Einsum2Plan::new(&[0u32], &[1u32], &[0u32, 1]).unwrap();
        assert!(plan.batch.is_empty());
        assert_eq!(plan.lo, vec![0]);
        assert_eq!(plan.ro, vec![1]);
        assert!(plan.sum.is_empty());
    }

    #[test]
    fn test_classify_dot_product() {
        // i,i->
        let plan = Einsum2Plan::new(&[0u32], &[0u32], &[] as &[u32]).unwrap();
        assert!(plan.batch.is_empty());
        assert!(plan.lo.is_empty());
        assert!(plan.ro.is_empty());
        assert_eq!(plan.sum, vec![0]);
    }

    #[test]
    fn test_classify_left_trace() {
        // ij,jk->k: lo=[], ro=[k], sum=[j], left_trace=[i]
        let plan = Einsum2Plan::new(&[0u32, 1], &[1u32, 2], &[2u32]).unwrap();
        assert!(plan.batch.is_empty());
        assert!(plan.lo.is_empty());
        assert_eq!(plan.ro, vec![2]);
        assert_eq!(plan.sum, vec![1]);
        assert_eq!(plan.left_trace, vec![0]);
    }

    #[test]
    fn test_perm_matmul() {
        // ij,jk->ik
        // A: [i, j] -> [batch=[], lo=[i], sum=[j]] = [i, j] => perm [0, 1]
        // B: [j, k] -> [batch=[], sum=[j], ro=[k]] = [j, k] => perm [0, 1]
        // C: [i, k] -> [batch=[], lo=[i], ro=[k]] = [i, k] => perm [0, 1]
        let plan = Einsum2Plan::new(&[0u32, 1], &[1u32, 2], &[0u32, 2]).unwrap();
        assert_eq!(plan.left_perm, vec![0, 1]);
        assert_eq!(plan.right_perm, vec![0, 1]);
        assert_eq!(plan.c_to_internal_perm, vec![0, 1]);
    }

    #[test]
    fn test_perm_batched_transposed_output() {
        // bij,bjk->bki (output has transposed lo/ro)
        let plan = Einsum2Plan::new(&[0u32, 1, 2], &[0u32, 2, 3], &[0u32, 3, 1]).unwrap();
        assert_eq!(plan.batch, vec![0]);
        assert_eq!(plan.lo, vec![1]);
        assert_eq!(plan.ro, vec![3]);
        assert_eq!(plan.sum, vec![2]);
        // C internal order: [b, i, k] = [0, 1, 3]
        // C IC order: [b, k, i] = [0, 3, 1]
        // c_to_internal_perm: maps IC positions to internal
        // internal[0]=b -> IC position 0
        // internal[1]=i -> IC position 2
        // internal[2]=k -> IC position 1
        assert_eq!(plan.c_to_internal_perm, vec![0, 2, 1]);
    }

    #[test]
    fn test_error_orphan_output() {
        let result = Einsum2Plan::new(&[0u32], &[1u32], &[0u32, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_duplicate() {
        let result = Einsum2Plan::new(&[0u32, 0], &[1u32], &[0u32, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_labels() {
        let plan = Einsum2Plan::new(&['i', 'j'], &['j', 'k'], &['i', 'k']).unwrap();
        assert_eq!(plan.lo, vec!['i']);
        assert_eq!(plan.ro, vec!['k']);
        assert_eq!(plan.sum, vec!['j']);
    }
}
