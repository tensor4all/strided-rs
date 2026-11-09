p = 'strided-basic/src/fuse/tests/tests.rs'
s = open(p, encoding='utf-8').read()
old = """    assert!(importance[1..].iter().enumerate().all(|(i, &w)| i == 1 && w == 0));"""
new = """    assert!(importance
        .iter()
        .enumerate()
        .all(|(i, &w)| i == 0 || i == 2 || w == 0));"""
assert old in s
open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print('ok')
