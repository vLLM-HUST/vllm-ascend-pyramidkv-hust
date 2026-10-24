# Benchmark originals

This bounded collection is preserved by the benchmark repository. Original versions, failures,
negative results and diagnostic scope remain unchanged; migration is not new runtime acceptance.
Other historical collections are not declared migrated by this change.

- [Original manifest 1](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9b81690f767aca56524ec6651535a9b8d484a4c3/archive/mod-git-20261022/pyramidkv-website-source/manifest.json)
- [Remaining current originals](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/manifest.json)

Together these collections preserve all 118 former `evidence/current/` records, including original
checksum-covered reports. The download combines both capsules at their original relative paths,
so internal historical references remain resolvable without changing or re-signing the originals.
The five `evidence/legacy/` records are now preserved in the [legacy manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/140a046d34ecf50bb8de3d8b19a7d70a6b0490e0/archive/mod-targeted-20261023/pyramid-legacy/manifest.json).
Legacy tests read the external originals and retain all SHA256, ratio and shared-provenance
assertions. Their EAGER/prefix-caching identity is historical, not current graph-mode admission.

Download and verify each original into an empty directory outside the source checkout:

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
export VLLM_HUST_BENCHMARK_ORIGINALS=/absolute/external/originals
```

The downloader checks the pinned manifest and each file's length and SHA256, rejects unsafe paths
and never executes archived code. CI repeats the readback and rejects the migrated paths returning
to Git. Fixed benchmark originals and unchanged source Git history preserve recoverability.

The [mixed-document manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5c970d6f36e42891d8365275fa6ec1b23333498b/archive/mod-targeted-20261023/pyramid-mixed/manifest.json) preserves the full
historical smoke and selected-page serving narratives, with LICENSE/NOTICE. Source documents now
retain method/installation guidance and link to these fixed original result bodies. This does not
change the legacy semantic tests, result identity or release acceptance boundary.

The [selected-page narrative manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4139441f928744af14f89c0cb95cd999cbd6915d/archive/mod-targeted-20261023/pyramid-selected/manifest.json)
preserves the full operator/serving document with LICENSE/NOTICE. Both this capsule and the
previous mixed-document capsule are now downloaded and SHA-checked. Three document projections
and five data-member links are checked against the external originals in CI. Current method and
reproduction guidance remains here; no historical measurement is promoted to new acceptance.
