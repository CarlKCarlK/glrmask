"""The experimental equivalence API stays usable without becoming public."""
import numpy as np
import pytest
import glrmask


@pytest.mark.parametrize("strategy", ["automatic", "compact", "dedicated"])
@pytest.mark.parametrize("constructor,source", [
    ("from_json_schema", '{"enum": [1, 2]}'),
    ("from_ebnf", 'start ::= "1" | "2"'),
    ("from_lark", 'start: "1" | "2"'),
    ("from_glrm", 'glrm 1; start start; nt start = "1" | "2";'),
])
def test_partition_preserves_duplicate_tokens_and_expands_sparse_ids(strategy, constructor, source):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"1", 1: b"2", 7: b"1", 31: b"3"})
    partition = getattr(glrmask._internal.VocabPartition, constructor)(source, vocab, strategy)
    class_id = partition.class_of(0)
    assert class_id is not None
    assert partition.class_of(7) == class_id
    members = partition.classes()[class_id]
    assert partition.representative(class_id) in members

    class_mask = [0] * partition.internal_mask_len
    word, bit = divmod(class_id, 64)
    class_mask[word] = 1 << bit
    expected = partition.expand_mask(class_mask)
    assert {i for i in range(32) if expected[i // 32] & (1 << (i % 32))} == set(members)
    actual = np.full(partition.original_mask_len, -1, dtype=np.int32)
    partition.fill_expanded_mask(class_mask, actual)
    assert actual.view(np.uint32).tolist() == expected


def test_partition_is_private_and_immutable():
    assert not hasattr(glrmask, "VocabPartition")
    assert not hasattr(glrmask, "VocabPartitionStrategy")
    assert glrmask._internal.VocabPartition.__module__ == "glrmask._internal"
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"1"})
    partition = glrmask._internal.VocabPartition.from_json_schema('{"const": 1}', vocab)
    with pytest.raises(AttributeError):
        partition.num_classes = 99


def test_partition_rejects_unknown_strategy():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"1"})
    with pytest.raises(ValueError, match="unknown vocab partition strategy"):
        glrmask._internal.VocabPartition.from_json_schema('{"const": 1}', vocab, "invalid")
