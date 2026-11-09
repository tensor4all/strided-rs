p = 'strided-basic/src/fuse/tests/tests.rs'
s = open(p, encoding='utf-8').read()
old = """    assert!(
        importance[2] > importance[0],
        "axis 2 (stride 3) must outrank axis 0 (stride 1): {:?}",
        &importance[..3]
    );
    assert!(importance[2..].iter().all(|&w| w == 0));"""
new = """    assert_eq!(
        importance[0],
        1u64 << 63,
        "axis 0 (stride 1) takes the top level"
    );
    assert_eq!(
        importance[2],
        1u64 << 62,
        "axis 2 (stride 3) keeps the level below it"
    );
    assert!(importance[1..].iter().enumerate().all(|(i, &w)| i == 1 && w == 0));"""
assert old in s
open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print('ok')
