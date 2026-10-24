# Selected-page serving result and acceptance boundary

[PR #13](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/13) avoids full V
gathers and reuses page mappings. Its operator improvement is not a demonstrated service
throughput benefit, HBM reduction, maximum-capacity qualification or release acceptance.

The [complete historical result and acceptance record](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5c970d6f36e42891d8365275fa6ec1b23333498b/archive/mod-targeted-20261023/pyramid-mixed/snapshot/docs/selected-page-serving-result.md)
preserves all original counts, positive/negative observations, exclusions, historical labels and
owner acceptance boundaries. The [canonical four-arm report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a2684c3701237c0b5a8cb6e73d25ce305d6f0e71/reports/pyramidkv-selected-page-abba-20261009)
contains the paired method, raw archives and qualification limits. Both arms enable compression;
this comparison must not be relabelled as the fixed official B0 versus optimized B1.

Implementation and reproduction instructions remain in this repository. Historical measured
results live in benchmark; migration does not change the original outcome or promote release
readiness. A corrected shared-adapter artifact still requires its own release-specific acceptance.
