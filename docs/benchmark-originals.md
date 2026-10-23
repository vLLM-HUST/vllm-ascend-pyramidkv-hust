# Benchmark originals

This bounded collection is preserved by the benchmark repository. Original versions, failures,
negative results and diagnostic scope remain unchanged; migration is not new runtime acceptance.
Other historical collections are not declared migrated by this change.

- [Original manifest 1](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9b81690f767aca56524ec6651535a9b8d484a4c3/archive/mod-git-20261022/pyramidkv-website-source/manifest.json)
- [Remaining current originals](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/manifest.json)

Together these collections preserve all 118 former `evidence/current/` records, including original
checksum-covered reports. The download combines both capsules at their original relative paths,
so internal historical references remain resolvable without changing or re-signing the originals.
The five `evidence/legacy/` records and their existing test consumers are not migrated by this change.

Download and verify each original into an empty directory outside the source checkout:

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
```

The downloader checks the pinned manifest and each file's length and SHA256, rejects unsafe paths
and never executes archived code. CI repeats the readback and rejects the migrated paths returning
to Git. Fixed benchmark originals and unchanged source Git history preserve recoverability.
