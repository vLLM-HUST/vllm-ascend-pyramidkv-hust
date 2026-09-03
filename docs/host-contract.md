# Required host contract

The recovered provider cannot safely activate through a generic import hook.
KV compression changes scheduler-visible physical ownership, so an active host
must supply an explicit transactional contract.

## Core responsibilities

The provider-neutral Core boundary must cover:

1. versioned configuration, compatibility, runtime-limit, and compression-plan
   values;
2. worker capability validation before formal KV allocation;
3. private destination admission for prefix-cached compression;
4. scheduler transport of plans, transaction identity, commit acknowledgement,
   cancellation, and restart handling;
5. atomic block-table replacement, ownership transfer, reference release, and
   physical-length accounting;
6. prefix-cache admission that preserves the provider-required recompute
   suffix; and
7. output serialization compatible with synchronous and asynchronous
   scheduling.

## Ascend responsibilities

The provider-neutral Ascend boundary must cover:

1. provider discovery without hard-coding PyramidKV in platform code;
2. pre-allocation worker validation and post-initialization activation;
3. a cache-write view that can select and materialize compact K/V only after a
   successful complete model forward;
4. request-local commit acknowledgement and layer-specific physical decode
   metadata; and
5. graph-safe metadata for validated decode-only capture modes.

The extension does not monkey-patch private model-runner or attention classes.
The paired host branches expose these responsibilities directly, so the active
manifest registers only the dedicated Ascend provider entry point and does not
claim the broad `vllm.general_plugins` namespace.
