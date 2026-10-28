# Downstream metadata adapter ownership

English | [简体中文](metadata-adapter-boundary.zh.md)

The [shared maintainer's request](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33#issuecomment-6460186520)
sets the direction: the shared plugin will expose generic overridable interfaces;
PyramidKV should own its specialized metadata implementation and state. Shared
PR #33 is Draft while that boundary is defined. Its current runtime diff is an
experimental reference, not a merge-ready implementation of the new boundary.

## Code to move downstream

The proposed `_metadata_tensor_cache`, `_batched_slot_storage`, correction-cache
policy, batched per-layer slot arithmetic and candidate flag belong with the
PyramidKV adapter. The adapter must also own its stable slot views and fresh
per-call length views. Method-specific MTP length/slot corrections should use
the generic metadata interface rather than introduce another shared dispatch
branch. Scheduler allocation, transaction acknowledgement and native lifecycle
ownership remain shared responsibilities.

The scope is the new optimization proposed in PR #33. Previously merged query,
per-layer state and graph integration remain the current host contract until
the upstream refactor supplies their replacement. This document does not
silently remove existing behavior or describe those older dependencies as
already eliminated.

## Interface requirements for the upstream design

The current `KVCompressionMethod` exposes cache binding, compression and query
observation. It does not expose an overridable metadata-construction callback;
the plugin directly instantiates its shared provider. Cache binding alone does
not let a downstream method replace per-layer attention metadata or the MTP
common view. A port cannot currently use only the public method contract.

The following are requirements for discussion, not an invented public API:

| Boundary | Needed contract |
| --- | --- |
| Binding before graph capture | Validated layer identity, cache geometry and token limits; downstream-owned address-stable storage |
| Each metadata build | Read-only committed physical anchors, current request ordering, positions, expanded block table and native metadata views |
| Returned views | Layer-keyed attention metadata and the associated speculative common view, including rejected-draft handling |
| Lifetime | Defined stream/lifetime guarantees, padded slots, request removal, cancellation, preemption, restart and cache rebinding |
| Compatibility and fallback | Explicit capability/version checks and a supported native continuation when the downstream layout is unsupported |

The upstream owner decides the final names, signatures and dispatch model.
Do not implement this by monkey-patching provider methods, changing an instance's
class, importing new private provider fields, or registering a competing host
hook. Those approaches would retain the dependency the refactor aims to remove.

## Port acceptance

After the generic contract is available, implement the adapter and its state in
this package, using only the agreed boundary. Keep the native path available for
unsupported layouts and leave the optimization opt-in until it is qualified.
Check reordered and mixed request rows, CPU/NPU metadata placement, expanded
pages, padding, immutable older length views, stable graph addresses, MTP
rollback and every reset/preemption lifecycle. Validate native slot equivalence
and live graph/speculative serving before making new performance claims.

The [frozen candidate and evidence](runtime-performance-boundary.md) remain tied
to their recorded source. Moving documents or defining an interface does not
transfer those receipts to new code. No new hardware experiment is implied by
this ownership-only change; any port needs its own qualification.
