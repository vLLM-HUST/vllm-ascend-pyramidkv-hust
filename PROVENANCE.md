# Provenance

Source archives:

- [intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831)
- [intellistream/vllm-ascend-hust-legacy-20260831](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831)

Primary history:

- [Core PR #232: transactional KV-compression lifecycle](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/232)
- [Ascend PR #225: PyramidKV provider](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/225)

The closed provider PR is migration input, not evidence of a released plugin.
Exact source commits, files, authors, tests, and benchmark receipts must be
recorded before alpha release.

## Historical anchors

| Role | Commit | Status |
| --- | --- | --- |
| Legacy Core PR #232 merge | `4861aab3af39e721c1b5a8b27b72c4f6bebda888` | Merged in the archived Core line |
| Legacy Core PR #232 head | `20ffb9bb9282222e9205375083e93e4d966d51cc` | Provider-neutral schema-v1 lifecycle |
| Legacy Core PR #238 merge | `bb5c59c8b3ca4a3e5dc05e55f3bf0622fcaf8dd3` | Merged CI repair in the archived Core line |
| Legacy Ascend PR #225 audit head | `13e695325072d7ac02578823541d18cbc325b8eb` | Closed, not released |
| Focused v2 base | `232d902d` | Archived vLLM-Ascend-HUST base |
| Focused v2 head | `a1d9f146f8327bb336f6348486c625a85155278f` | Migration source |

The focused v2 series was authored and signed off by Mhyzb
(`<2372717433@qq.com>`):

1. `df8526b16ce3793a1bdfa6a630f1718ffc117573` — provider integration
2. `95bc201fb985f5a50625e81eb131e805365d0f39` — integration tests
3. `a12612f5000dd98fe792e3a2b84d5e2c82fae82a` — user documentation
4. `a1d9f146f8327bb336f6348486c625a85155278f` — paired CI validation

The preserved `final` and `v2` branches are frozen audit and migration inputs.
They are not installation sources and should not be deleted or renamed before
an independently reproducible alpha is tagged here.

## Recovered file mapping

The initial extraction uses the identical provider blob present at both the
legacy PR audit head and the focused v2 head, then changes only imports and
type boundaries needed to make the new package independently importable.

| Legacy path at v2 head | Git blob | New path | Treatment |
| --- | --- | --- | --- |
| `vllm_ascend/kv_cache_compression/pyramidkv.py` | `7b361031cbbde258b13d50142f3a817c1ee897c5` | `src/vllm_ascend_pyramidkv/provider.py` | Recovered; host imports isolated and contracts redirected to a frozen local mirror |
| `vllm_ascend/kv_cache_compression/registry.py` | `88b1e99d008ee10034633b73862b064ff858980f` | `src/vllm_ascend_pyramidkv/registry.py` | Rewritten as a side-effect-free lazy registry |
| `docs/source/user_guide/feature_guide/pyramidkv.md` | `c5dac7846f90301e424cdb89d064bdb9e10da455` | `docs/` and `README.md` | Claims downgraded to historical/pending where current-head evidence is absent |
| `tests/ut/kv_cache_compression/test_pyramidkv.py` | `385371776af02a359f94b48cd5f43fe32fa9c832` | `tests/` | CPU-safe algorithm, compatibility, state, and registry semantics restored |
| `tests/ut/kv_cache_compression/test_registry.py` | `d776b7e7f7aab40778febce85c6d553c8ed1b266` | `tests/test_registry.py` | Registry fail-closed semantics restored |

The frozen schema-v1 data classes in `legacy_contracts.py` derive from the
Apache-2.0 provider-neutral contract in Core PR #232. They exist only for
migration testing and do not assert that the current host exports that API.

The initial extracted and formatted `provider.py` has SHA-256
`949e6a2f612618e19b35372402ce140bb0f3fced8a3401e3a4854df6ba210c16`.
Future changes must update the evidence mapping rather than overwriting the
legacy blob identity above.

## Algorithm references

- [PyramidKV paper](https://arxiv.org/abs/2406.02069)
- [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory), MIT License
- [Irisuko/KVCache-Factory-Ascend at `fc6f8f4c`](https://github.com/Irisuko/KVCache-Factory-Ascend/commit/fc6f8f4c3d8ca7a1849a2ef67ff5fca8d285a6f0), the exact Ascend-fork commit cited by the recovered provider

The recovered provider states that it is a clean-room implementation of
observable selection semantics and does not import or copy the reference
repository's Hugging Face patches, CUDA, Triton, or custom kernels. See
`NOTICE` for attribution and licensing notes.

## Evidence policy

Historical 910B2, LongBench, and online-serving artifacts remain legacy
supporting evidence. They must not be presented as results of this repository's
current head. A runnable alpha requires a clean installed artifact, an exact
host/provider commit pair, raw real-device evidence, and verified disable,
restart, uninstall, and rollback behavior.
