use super::*;

fn wire(lengths: [usize; 11]) -> Vec<u8> {
    let mut bytes = SECTION_MAGIC.to_vec();
    for length in lengths {
        bytes.extend_from_slice(&(length as u64).to_le_bytes());
    }
    for (index, length) in lengths.into_iter().enumerate() {
        bytes.extend(std::iter::repeat_n(index as u8, length));
    }
    bytes
}

fn parts(sections: ConstraintSections<'_>) -> [&[u8]; 11] {
    [sections.weight, sections.dwa, sections.table, sections.core,
     sections.runtime, sections.token_bytes, sections.original_map,
     sections.tokenizer, sections.internal_masks, sections.token_mask_cache,
     sections.composition_metadata]
}

#[test]
fn named_sections_follow_writer_order_without_copying() {
    let lengths = [0, 1, 2, 0, 4, 0, 3, 5, 0, 2, 1];
    let bytes = wire(lengths);
    let sections = parts(constraint_sections(&bytes).unwrap());
    let mut offset = SECTION_HEADER_LEN;
    for (index, section) in sections.into_iter().enumerate() {
        assert_eq!(section, vec![index as u8; lengths[index]]);
        assert_eq!(section.as_ptr(), bytes[offset..].as_ptr());
        offset += lengths[index];
    }
    assert_eq!(offset, bytes.len());
}

#[test]
fn empty_sections_still_borrow_the_end_of_the_header() {
    let bytes = wire([0; 11]);
    for section in parts(constraint_sections(&bytes).unwrap()) {
        assert!(section.is_empty());
        assert_eq!(section.as_ptr(), bytes[SECTION_HEADER_LEN..].as_ptr());
    }
}

#[test]
fn every_truncated_prefix_is_rejected() {
    let bytes = wire([1; 11]);
    for end in 0..bytes.len() {
        assert!(constraint_sections(&bytes[..end]).is_err(), "accepted prefix {end}");
    }
    assert!(constraint_sections(&bytes).is_ok());
}

#[test]
fn overflowing_lengths_are_rejected_in_every_section() {
    for index in 0..11 {
        let mut bytes = wire([0; 11]);
        let start = SECTION_MAGIC.len() + index * 8;
        bytes[start..start + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(constraint_sections(&bytes).is_err(), "accepted overflow at {index}");
    }
}

#[test]
fn lengths_cannot_overrun_the_backing_slice() {
    let mut bytes = wire([0; 11]);
    bytes[4..12].copy_from_slice(&1_u64.to_le_bytes());
    assert!(constraint_sections(&bytes).is_err());
}

#[test]
fn trailing_payload_is_rejected() {
    let mut bytes = wire([1; 11]);
    bytes.push(0);
    assert!(constraint_sections(&bytes).is_err());
}

#[test]
fn isolated_and_historical_section_versions_are_rejected() {
    let bytes = wire([0; 11]);
    for magic in [b"S30\0", b"S31\0", b"S32\0", b"nope"] {
        let mut old = bytes.clone();
        old[..4].copy_from_slice(magic);
        assert!(constraint_sections(&old).is_err());
    }
    assert_eq!(CONSTRAINT_VERSION, 33);
    assert_eq!(&SECTION_MAGIC, b"S33\0");
}
