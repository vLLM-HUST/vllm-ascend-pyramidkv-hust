# Required shared-host method contract

The former direct Core and Ascend host Drafts were withdrawn. PyramidKV must
not patch those source trees or reactivate their private contract. The confirmed
shared lifecycle owner is `vllm-ascend-kvcompress-hust`, with activation and
rollback controlled through Extension Manager.

## Existing shared boundary

The shared package documents the external entry-point group
`vllm_ascend_kvcompress.methods`. Its method API already provides:

1. model shape and method-owned configuration;
2. scheduler-visible runtime limits;
3. allocated full-attention K/V layer bindings;
4. source and private destination block tables; and
5. synchronous compression results, including an optional per-layer length
   result.

## Accepted query observation

PyramidKV scores historical keys with the final prefill's trailing query
window. Shared-host PR #9 added an optional observation contract that:

1. remains a no-op for methods that do not request it;
2. declares the trailing query-window length before allocation;
3. stages only bound full-attention layers across prefill chunks;
4. binds observations to request and compression-transaction identity;
5. publishes them only after the complete final-prefill forward and sampling
   succeed; and
6. clears state on cancellation, abort, restart, and completed commit.

PyramidKV accepts that contract. Earlier chunk staging is necessary to retain a
window that crosses chunk boundaries; uncommitted observations remain invisible
to compression.

## Accepted per-layer physical state

PyramidKV assigns different retained lengths to different layers. Shared-host
PR #10 validates the complete `per_layer_physical_num_tokens` map, commits it
at the output-acknowledged transaction boundary, and applies each layer's
anchor to eager decode slots and standard Ascend attention metadata. Uniform
methods keep their existing path; GDN remains in semantic space.

Merged shared-host PR #13 lets a method declare unequal per-layer state
before graph capture, allocates stable layer-specific slot buffers, and uses
the host's layer-keyed FIA task update seam during `FULL_AND_PIECEWISE` replay.
It also permits a query method to select target-model layers while retaining an
auxiliary MTP cache in the materialized result. The MTP speculative common
metadata receives that cache's physical lengths and slots, so rejected draft
positions are overwritten rather than advancing per-layer state. Unsupported
graph/speculative/parallel paths remain fail closed.

The PyramidKV package CI pins the merged shared-host contract exactly at
`a82e08798c10cee8bc0f0468fb0b05def064ae78` (release PR #14). This
versions the shared method API as `METHOD_API_VERSION = 1` and the package as
`0.9.0`; PyramidKV requires `>=0.9,<0.10`. Source-based CI does not establish
availability from the approved package source. Verify the published package
before merging activation or running release qualification.

## Accepted prefix-cache admission

The public runtime spec declares `required_recompute_tokens`, but the shared
scheduler now applies it to prefix-cache admission. PyramidKV needs the last
`window_size` query rows; shared-host PR #12 caps eligible cache hits at
`prompt_len - required_recompute_tokens`, with a zero lower bound for short
prompts. It also rejects an external method whose query window exceeds its
declared recompute requirement and fails closed when APC lookup hooks are
missing. PyramidKV accepts this contract.

The shared provider must contain no PyramidKV-specific policy. Design
coordination remains in
https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/3.

## Target acceptance boundary

Interface acceptance alone is not runtime support. Activation is restricted to
CANN 9.1 and the official `Qwen/Qwen3.5-35B-A3B` model (display name:
Qwen3.5-35B), using the exact validated TP=2 profile with APC, MTP=2, async
scheduling, `FULL_AND_PIECEWISE`, and `mamba_cache_mode=align` enabled. Other
profiles fail closed. Only full-attention K/V may be compacted; recurrent state
stays native.
