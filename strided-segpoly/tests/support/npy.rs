//! Minimal reader for the `.npy` files the fixture generator emits.
//!
//! Handles what `numpy.save` writes for a C-ordered `float64` or `int32` array:
//! format version 1 or 2, `fortran_order: False`, and a tuple `shape`. Anything
//! else is an error rather than a silent reinterpretation — `strided-rs` has no
//! `numpy` dependency, and the fixtures are generated, so a wrong file must fail
//! loudly rather than produce a plausible result.

use std::path::Path;

/// What a header says about the payload.
struct Header {
    shape: Vec<usize>,
    descr: String,
    offset: usize,
}

/// Number of elements of `shape`.
fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Read a `float64` array, asserting the shape the case was generated with.
pub fn read_f64(path: impl AsRef<Path>, shape: &[usize]) -> Vec<f64> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let header = parse_header(&bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(header.shape.as_slice(), shape, "{}: shape", path.display());
    assert!(
        matches!(header.descr.as_str(), "<f8" | "|f8" | "=f8"),
        "{}: expected float64, got {}",
        path.display(),
        header.descr
    );
    let payload = &bytes[header.offset..];
    let len = element_count(shape);
    assert_eq!(payload.len(), len * 8, "{}: payload length", path.display());
    payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| f64::from_le_bytes(*word))
        .collect()
}

/// Read an `int32` array, asserting the shape the case was generated with.
pub fn read_i32(path: impl AsRef<Path>, shape: &[usize]) -> Vec<i32> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let header = parse_header(&bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(header.shape.as_slice(), shape, "{}: shape", path.display());
    assert!(
        matches!(header.descr.as_str(), "<i4" | "|i4" | "=i4"),
        "{}: expected int32, got {}",
        path.display(),
        header.descr
    );
    let payload = &bytes[header.offset..];
    let len = element_count(shape);
    assert_eq!(payload.len(), len * 4, "{}: payload length", path.display());
    payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect()
}

fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if !bytes.starts_with(MAGIC) {
        return Err("missing .npy magic".into());
    }
    let major = *bytes.get(MAGIC.len()).ok_or("truncated version")?;
    let rest = &bytes[MAGIC.len() + 2..];
    let (header_len, after) = match major {
        1 => {
            let len = usize::from(u16::from_le_bytes(
                rest[..2]
                    .try_into()
                    .map_err(|_| "truncated header length")?,
            ));
            (len, &rest[2..])
        }
        2 | 3 => {
            let len = u32::from_le_bytes(
                rest[..4]
                    .try_into()
                    .map_err(|_| "truncated header length")?,
            ) as usize;
            (len, &rest[4..])
        }
        other => return Err(format!("unsupported .npy version {other}")),
    };
    let header = core::str::from_utf8(after.get(..header_len).ok_or("header runs past the file")?)
        .map_err(|_| "header is not UTF-8".to_string())?;
    if !header.contains("'fortran_order': False") {
        return Err(format!("expected a C-order array, header: {header}"));
    }
    let descr = field(header, "descr")?;
    let shape = field(header, "shape")?
        .trim_matches(['(', ')'])
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("bad shape: {error}"))?;
    // The payload starts after the magic, the version, the length field and the
    // header: whatever `after` skipped, plus the header itself.
    let prefix = bytes.len() - after.len();
    Ok(Header {
        shape,
        descr,
        offset: prefix + header_len,
    })
}

/// The text after `'name':` in the header dict, up to the closing quote/paren.
fn field(header: &str, name: &str) -> Result<String, String> {
    let key = format!("'{name}':");
    let start = header
        .find(&key)
        .ok_or_else(|| format!("header lacks {key}"))?
        + key.len();
    let rest = header[start..].trim_start();
    let value = match rest.strip_prefix(['"', '\'']) {
        Some(quoted) => quoted
            .split(['"', '\''])
            .next()
            .ok_or("unterminated string")?
            .to_string(),
        None if rest.starts_with('(') => {
            // A tuple value: take through its closing paren, not to the next comma.
            let inner = &rest[1..];
            let end = inner.find(')').ok_or("unterminated tuple")?;
            format!("({})", &inner[..end])
        }
        None => rest
            .split(['}', ','])
            .next()
            .unwrap_or("")
            .trim()
            .to_string(),
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let mut padded = header.to_string();
        while !(10 + padded.len() + 1).is_multiple_of(64) {
            padded.push(' ');
        }
        padded.push('\n');
        bytes.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        bytes.extend_from_slice(padded.as_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn parses_a_float64_image() {
        let payload: Vec<u8> = (0..6u8).flat_map(|v| f64::from(v).to_le_bytes()).collect();
        let bytes = image(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }",
            &payload,
        );
        let header = parse_header(&bytes).expect("valid image");
        assert_eq!(header.shape, vec![2, 3]);
        assert_eq!(header.offset, bytes.len() - 48);
        let values: Vec<f64> = bytes[header.offset..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| f64::from_le_bytes(*word))
            .collect();
        assert_eq!(values, (0..6).map(f64::from).collect::<Vec<_>>());
    }

    #[test]
    fn rejects_a_fortran_order_image() {
        let bytes = image(
            "{'descr': '<f8', 'fortran_order': True, 'shape': (2,), }",
            &[0; 16],
        );
        assert!(parse_header(&bytes).is_err());
    }
}
