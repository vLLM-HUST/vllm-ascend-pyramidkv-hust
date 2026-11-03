# SPDX-License-Identifier: Apache-2.0
"""Contract checks against the installed upstream-aligned host packages."""

from __future__ import annotations

import inspect
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from packaging.specifiers import SpecifierSet

pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")


def _parameter_names(callable_object: object) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def test_installed_host_versions_are_in_the_validated_lines() -> None:
    core_version = version("vllm")
    ascend_version = version("vllm-ascend")
    assert core_version == "0.23.0+empty" or core_version in SpecifierSet(">=0.29.1.post1.dev0,<0.30")
    assert ascend_version == "0.23.0.post1" or ascend_version in SpecifierSet(">=0.25.1rc2.dev0,<0.26")
    assert version("vllm-hust-ext") in SpecifierSet(">=0.2.0.dev0,<0.4")


def test_scheduler_and_cache_manager_seams_match_current_host() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    assert _parameter_names(Scheduler.schedule) in {
        ("self",),
        ("self", "throttle_prefills"),
    }
    assert _parameter_names(KVCacheManager.allocate_slots)[:3] == (
        "self",
        "request",
        "num_new_tokens",
    )
    assert _parameter_names(KVCacheManager.free) == ("self", "request")


def test_hybrid_prefill_requires_budget_beyond_one_aligned_block():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=2048), use_eagle=False)
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=7168, num_tokens=7168)
    # Real host characterization: a decode batch leaves no complete prefill
    # block when the entire token budget equals one Mamba alignment block.
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 2047) == 0
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 2016) == 0
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 4095) == 2048
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 4064) == 2048


def test_ascend_v1_worker_seams_match_current_host() -> None:
    try:
        from vllm_ascend.worker.block_table import BlockTable, MultiGroupBlockTable
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except (ImportError, OSError, RuntimeError, SystemError) as error:
        pytest.skip(f"Ascend runtime is unavailable in this test environment: {error}")

    assert _parameter_names(BlockTable.compute_slot_mapping) == (
        "self",
        "num_reqs",
        "query_start_loc",
        "positions",
    )
    assert _parameter_names(BlockTable.add_row) == ("self", "block_ids", "row_idx")
    assert hasattr(MultiGroupBlockTable, "block_tables") is False
    assert _parameter_names(MultiGroupBlockTable.add_row) == (
        "self",
        "block_ids",
        "row_idx",
    )
    for method in (
        "initialize_kv_cache",
        "_update_states",
        "_build_attention_metadata",
        "sample_tokens",
    ):
        assert callable(getattr(NPUModelRunner, method))


def test_grouped_topk_router_scaling_compatibility() -> None:
    from vllm.model_executor.layers.fused_moe.config import (
        RoutingMethodType,
        get_routing_method_type,
    )

    from vllm_ascend_pyramidkv.adapter.host_compat import install_grouped_topk_router_compat

    if "routed_scaling_factor" in _parameter_names(get_routing_method_type):
        assert install_grouped_topk_router_compat() is False
        classify = get_routing_method_type
    else:
        from vllm_ascend.ops.fused_moe.router import grouped_topk_router

        install_grouped_topk_router_compat()
        classify = grouped_topk_router.get_routing_method_type
    options = dict(
        scoring_func="sigmoid",
        top_k=8,
        renormalize=True,
        num_expert_group=None,
        has_e_score_bias=True,
    )
    assert classify(**options, routed_scaling_factor=1.0) == RoutingMethodType.MiniMax2
    assert classify(**options, routed_scaling_factor=2.0) == RoutingMethodType.Unspecified
    assert install_grouped_topk_router_compat() is False
