"""Exercise both packed coordinates without serializing large mask traces."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native


@pytest.mark.parametrize("strategy", ("automatic", "compact", "dedicated"))
def test_partition_expands_u64_class_boundaries_to_full_width_model_words(strategy):
    # In a fixed ordered sequence, each distinct word has a different valid
    # parser position. These tokens must not share an equivalence class.
    words = [f"a{i:03d}".encode() for i in range(80)]
    tokens = dict(enumerate(words))
    tokens.update({2097151: words[63], 2097152: words[64]})
    vocabulary = g.Vocab.from_id_to_bytes(tokens)
    grammar = "start ::= " + " ".join('"' + word.decode() + '"' for word in words)
    partition = native._internal.VocabPartition.from_ebnf(
        grammar, vocabulary, strategy=strategy
    )
    classes = partition.classes()
    assert sorted(token for group in classes for token in group) == sorted(tokens)
    assert len({partition.class_of(token) for token in range(80)}) == 80
    assert partition.class_of(2097151) == partition.class_of(63)
    assert partition.class_of(2097152) == partition.class_of(64)
    assert len(classes) >= 80

    # Class-space words use u64, whereas model-space output words use u32.
    # The last alias is in output word 65536, beyond the former u16 extent.
    width = partition.original_mask_len + 2
    storage = np.full(width + 4, -314159, dtype=np.int32)
    output = storage[2:-2]
    selections = (
        list(range(len(classes))),
        [],
        [63],
        [64],
        [63, 64],
        [len(classes) - 1],
        [],
    )
    for selected in selections:
        packed = [0] * ((len(classes) + 63) // 64 + 1)
        for class_id in selected:
            packed[class_id // 64] |= 1 << (class_id % 64)
        packed[-1] = (1 << 64) - 1  # Nonexistent classes must be ignored.

        expected = np.zeros(width, dtype=np.uint32)
        for class_id in selected:
            for token in classes[class_id]:
                expected[token // 32] |= np.uint32(1 << (token % 32))
        partition.fill_expanded_mask(packed, output)
        np.testing.assert_array_equal(output.view(np.uint32), expected)
        np.testing.assert_array_equal(storage[[0, 1, -2, -1]], [-314159] * 4)
        np.testing.assert_array_equal(
            np.asarray(partition.expand_mask(packed), dtype=np.uint32),
            expected[:partition.original_mask_len],
        )
