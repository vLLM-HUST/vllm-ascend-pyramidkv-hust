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
PR #10 now validates the complete `per_layer_physical_num_tokens` map, commits
it at the output-acknowledged transaction boundary, and applies each layer's
anchor to eager decode slots and standard Ascend attention metadata. Uniform
methods keep their existing path; GDN remains in semantic space. Graph replay,
MTP, and other metadata backends remain fail closed.

## Remaining prefix-cache admission gap

The public runtime spec declares `required_recompute_tokens`, but the shared
scheduler does not yet apply it to prefix-cache admission. PyramidKV needs the
last `window_size` query rows; an APC hit that leaves only one uncached token
cannot satisfy the observation contract. The external method therefore rejects
APC until the scheduler caps cache hits at
`prompt_len - required_recompute_tokens`, with a zero lower bound for short
prompts.

The shared provider must contain no PyramidKV-specific policy. Design
coordination remains in
https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/3.

## Target acceptance boundary

Interface acceptance is not runtime support. CANN 9.1 and
Qwen3.5-35B-A3B hybrid serving must continue to fail closed until an exact
package trio validates TP=2 with APC, MTP=2, async scheduling,
`FULL_AND_PIECEWISE`, and `mamba_cache_mode=align` enabled. Only
full-attention K/V may be compacted; recurrent state stays native.
