# PyramidKV experimental protocols

English | [简体中文](README.zh.md)

PyramidKV owns these method-specific protocols. The shared compression plugin
owns common lifecycle and extension contracts. Measured results, logs, profiles
and original experiment contracts remain in `vllm-hust-benchmark`.

- [Metadata and service net cost](pyramidkv-net-cost.md)
- [Fixed-budget memory pressure](pyramidkv-memory-pressure.md)
- [Metadata adapter ownership and migration](../metadata-adapter-boundary.md)

The four protocol files were copied byte-for-byte from shared commit
[`b6e9836b97955be923c26d6565a7ccf888523fa4`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/tree/b6e9836b97955be923c26d6565a7ccf888523fa4/docs/benchmark)
following the [maintainer's ownership request](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33#issuecomment-6460186520).
Moving the maintained copies does not change historical preregistration or
measured-source identities. Existing fixed-commit links remain valid history.

The batched-source flag described in the protocols belongs to the frozen
experimental candidate. It does not imply that a downstream metadata adapter
has been integrated against the forthcoming generic interface.

Use the [canonical evidence index](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c4093ae07d49ced6ac534befb62c10498ccbd08f/reports/pyramidkv-metadata-net-cost-20261028/README.md)
for observations and original bytes. Source ownership changes are not new
runtime, performance, quality or public-wheel acceptance.
