use std::mem::MaybeUninit;
use strided_basic::{copy_into_uninit, StridedError, StridedView, StridedViewMut};

fn check(
    dims: &[usize],
    source_strides: &[isize],
    source_offset: isize,
    dest_strides: &[isize],
    dest_offset: isize,
    extent: usize,
) {
    let source: Vec<f64> = (0..extent).map(|i| i as f64 - 13.0).collect();
    let mut output = vec![MaybeUninit::new(-999.0); extent];
    let mut expected = vec![-999.0; extent];
    let count: usize = dims.iter().product();
    for linear in 0..count {
        let mut index = linear;
        let (mut src, mut dst) = (source_offset, dest_offset);
        for ((&dim, &ss), &ds) in dims.iter().zip(source_strides).zip(dest_strides) {
            let coordinate = (index % dim) as isize;
            index /= dim;
            src += coordinate * ss;
            dst += coordinate * ds;
        }
        expected[dst as usize] = source[src as usize];
    }
    copy_into_uninit(
        &mut StridedViewMut::new(&mut output, dims, dest_strides, dest_offset).unwrap(),
        &StridedView::new(&source, dims, source_strides, source_offset).unwrap(),
    )
    .unwrap();
    // SAFETY: all slots started initialized, including holes.
    let values: Vec<_> = output
        .into_iter()
        .map(|x| unsafe { x.assume_init() })
        .collect();
    assert_eq!(values, expected);
}

#[test]
fn layouts_and_holes_match_reference() {
    check(&[], &[], 2, &[], 3, 10);
    check(&[0, 3], &[1, 0], 0, &[1, 0], 0, 0);
    check(&[3, 4], &[1, 3], 1, &[1, 3], 2, 20);
    check(&[3, 4], &[4, 1], 1, &[1, 3], 2, 20);
    check(&[3, 4], &[-1, 5], 3, &[2, 9], 7, 50);
    check(&[3, 4], &[1, 3], 0, &[-1, 3], 2, 12);
    check(&[4, 5], &[0, 1], 0, &[1, 4], 0, 20);
    check(&[2, 3, 4], &[12, 1, 3], 0, &[1, 2, 6], 0, 24);
    check(&[1; 12], &[1; 12], 0, &[1; 12], 0, 1);
}

#[test]
fn broadcast_batches_match_reference() {
    check(&[3, 2, 2], &[1, 3, 0], 0, &[1, 3, 6], 0, 12);
    check(&[2, 3, 2], &[0, 1, 3], 0, &[1, 2, 6], 0, 12);
    check(&[2, 2, 3], &[2, 0, 4], 0, &[1, 2, 4], 0, 12);
}

#[test]
fn copy_preserves_complex_bits_and_initializes_all_slots() {
    use num_complex::Complex64;
    let source = [
        Complex64::new(f64::from_bits(0x7ff8000000000042), -0.0),
        Complex64::new(f64::INFINITY, f64::NEG_INFINITY),
    ];
    let mut output = [MaybeUninit::uninit(); 2];
    copy_into_uninit(
        &mut StridedViewMut::new(&mut output, &[2], &[1], 0).unwrap(),
        &StridedView::new(&source, &[2], &[-1], 1).unwrap(),
    )
    .unwrap();
    for (dst, src) in output.iter().zip(source.iter().rev()) {
        // SAFETY: every destination element was written by the successful copy.
        let actual = unsafe { dst.assume_init() };
        assert_eq!(actual.re.to_bits(), src.re.to_bits());
        assert_eq!(actual.im.to_bits(), src.im.to_bits());
    }
}

#[test]
fn invalid_destination_is_unchanged_and_zst_is_supported() {
    let source = [1.0; 4];
    let mut output = [MaybeUninit::new(-1.0); 4];
    let err = copy_into_uninit(
        &mut StridedViewMut::new(&mut output, &[2, 2], &[0, 1], 0).unwrap(),
        &StridedView::new(&source, &[2, 2], &[1, 2], 0).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(err, StridedError::NonInjectiveOutputLayout));
    for slot in output {
        assert_eq!(unsafe { slot.assume_init() }, -1.0);
    }
    let mut zst = [MaybeUninit::uninit(); 6];
    copy_into_uninit(
        &mut StridedViewMut::new(&mut zst, &[2, 3], &[1, 2], 0).unwrap(),
        &StridedView::new(&[(); 6], &[2, 3], &[3, 1], 0).unwrap(),
    )
    .unwrap();
}

#[test]
fn tiled_float_copies_preserve_bits_and_padded_types_fall_back() {
    macro_rules! floats {
        ($ty:ty, $nan:expr) => {{
            let values: [$ty; 4] = [$nan, -0.0, <$ty>::INFINITY, <$ty>::NEG_INFINITY];
            let source: Vec<_> = (0..72).map(|i| values[i % 4]).collect();
            let mut output = vec![MaybeUninit::uninit(); 72];
            copy_into_uninit(
                &mut StridedViewMut::new(&mut output, &[8, 9], &[1, 8], 0).unwrap(),
                &StridedView::new(&source, &[8, 9], &[9, 1], 0).unwrap(),
            )
            .unwrap();
            for col in 0..9 {
                for row in 0..8 {
                    // SAFETY: the successful copy initialized the entire output.
                    let value = unsafe { output[row + 8 * col].assume_init() };
                    assert_eq!(value.to_bits(), source[9 * row + col].to_bits());
                }
            }
        }};
    }
    floats!(f32, f32::from_bits(0x7fc00042));
    floats!(f64, f64::from_bits(0x7ff8000000000042));

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Padded(u8, u32);
    let source = [Padded(1, 11), Padded(2, 22), Padded(3, 33), Padded(4, 44)];
    let mut output = [MaybeUninit::uninit(); 4];
    copy_into_uninit(
        &mut StridedViewMut::new(&mut output, &[2, 2], &[1, 2], 0).unwrap(),
        &StridedView::new(&source, &[2, 2], &[2, 1], 0).unwrap(),
    )
    .unwrap();
    for (i, expected) in [source[0], source[2], source[1], source[3]]
        .iter()
        .enumerate()
    {
        // SAFETY: the successful copy initialized each logical Padded value.
        assert_eq!(unsafe { output[i].assume_init() }, *expected);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn large_permutation_respects_sequential_and_bounded_policy() {
    use std::num::NonZeroUsize;
    use strided_basic::{with_execution_policy, ExecutionPolicy};
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            for policy in [
                ExecutionPolicy::Sequential,
                ExecutionPolicy::Rayon {
                    max_threads: NonZeroUsize::new(2).unwrap(),
                },
            ] {
                with_execution_policy(policy, || {
                    check(
                        &[12, 12, 1100],
                        &[1100, 13200, 1],
                        0,
                        &[1, 12, 144],
                        0,
                        158400,
                    );
                });
            }
        });
}
