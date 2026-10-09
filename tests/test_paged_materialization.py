"""Compare selected-page copying against the original full-prefix algorithm."""

import pytest
import torch
from vllm_ascend_kvcompress.methods.base import LayerCache

from vllm_ascend_pyramidkv.method import (
    _copy_selected_kv,
    _gather_paged_prefix,
    _paged_slots,
    _selected_cache_rows,
)
from vllm_ascend_pyramidkv.provider import materialize_pyramid_kv


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("tokens,retained,window", [(4097, 190, 4), (8192, 2048, 8), (511, 129, 8)])
def test_selected_pages_match_full_materialization(heads, tokens, retained, window):
    torch.manual_seed(17)
    source_count = (tokens + 127) // 128
    destination_count = (retained + 127) // 128 + 1
    blocks = source_count + destination_count + 2
    permutation = torch.randperm(blocks)
    source = permutation[:source_count].to(torch.int32)
    destination = permutation[source_count : source_count + destination_count].to(torch.int32)
    key = torch.randn(blocks, 128, heads, 8)
    value = torch.randn_like(key)
    before = (key.clone(), value.clone())
    selected = torch.stack([torch.randperm(tokens - window)[: retained - window] for _ in range(heads)]).unsqueeze(0)
    full = [_gather_paged_prefix(cache, source, tokens).permute(1, 0, 2).unsqueeze(0) for cache in before]
    compact = materialize_pyramid_kv(*full, selected, retained, window)
    # Independent destination mapping, with all padding and unrelated pages
    # included in the final whole-cache comparison.
    expected = [cache.clone() for cache in before]
    for result, values in zip(expected, compact, strict=True):
        for position in range(retained):
            result[destination[position // 128], position % 128] = values[0, :, position]
    source_slots = _paged_slots(source, tokens, key.device)
    rows = _selected_cache_rows(source_slots, selected, window)
    destination_slots = _paged_slots(destination, destination_count * 128, key.device)
    _copy_selected_kv(LayerCache("test", 0, key, value), rows, destination_slots[:retained])
    assert torch.equal(key, expected[0])
    assert torch.equal(value, expected[1])
    assert torch.equal(key[source.long()], before[0][source.long()])
    assert torch.equal(value[source.long()], before[1][source.long()])


def test_short_page_table_rejected_before_materialization():
    with pytest.raises(RuntimeError, match="too short"):
        _paged_slots(torch.tensor([7]), 129, torch.device("cpu"))
