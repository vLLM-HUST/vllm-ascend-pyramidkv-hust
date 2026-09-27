# Required shared-host method contract

The former direct Core and Ascend host Drafts were withdrawn. PyramidKV must
not patch those source trees or reactivate their private contract. The proposed
shared lifecycle owner is `vllm-ascend-kvcompress-hust`, with activation and
rollback controlled through Extension Manager.

## Existing shared boundary

The shared package documents the external entry-point group
`vllm_ascend_kvcompress.methods`. Its method API already provides:

1. model shape and method-owned configuration;
2. scheduler-visible runtime limits;
3. allocated full-attention K/V layer bindings;
4. source and private destination block tables; and
5. synchronous compression results and physical-length accounting.

## Missing PyramidKV input

PyramidKV scores historical keys with the final prefill's trailing query
window. The current method API supplies K/V caches but does not expose those
query tensors. An external method therefore cannot reproduce PyramidKV token
selection without independently patching private model-runner or attention
classes, which this repository will not do.

The required optional query-observation extension must:

1. remain a no-op for methods that do not request it;
2. declare the trailing query-window length before allocation;
3. observe only full-attention layers after a successful final-prefill forward;
4. bind observations to request and compression-transaction identity;
5. use method-owned, address-stable buffers for validated graph replay; and
6. clear state on cancellation, abort, restart, and completed commit.

The shared provider must contain no PyramidKV-specific policy. The algorithm
will register as an external method only after this interface is accepted.
Design coordination is tracked in
https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/3.

## Target acceptance boundary

Interface acceptance is not runtime support. CANN 9.1 and
Qwen3.5-35B-A3B hybrid serving must continue to fail closed until an exact
package trio validates TP=2 with APC, MTP=2, async scheduling,
`FULL_AND_PIECEWISE`, and `mamba_cache_mode=align` enabled. Only
full-attention K/V may be compacted; recurrent state stays native.
