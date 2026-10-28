# PyramidKV 实验规程

[English](README.md) | 简体中文

PyramidKV 下游仓库负责这些方法专用规程；共享压缩插件负责通用生命周期和扩展契约。
实测结果、日志、profile 和原始实验合同仍统一保存在 `vllm-hust-benchmark`。

- [元数据与服务净成本](pyramidkv-net-cost.zh.md)
- [固定预算内存压力](pyramidkv-memory-pressure.zh.md)
- [元数据适配器归属与迁移](../metadata-adapter-boundary.zh.md)

根据[维护者的分工要求](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33#issuecomment-6460186520)，
四份规程从共享仓库提交
[`b6e9836b97955be923c26d6565a7ccf888523fa4`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/tree/b6e9836b97955be923c26d6565a7ccf888523fa4/docs/benchmark)
逐字节迁入。维护位置变化不改变历史预登记及实测源码身份，既有固定提交链接保留为历史依据。

规程中的批量元数据开关属于冻结的实验候选，不表示下游元数据适配器已经接入未来的通用接口。
观测结果和原件见[权威证据索引](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/c4093ae07d49ced6ac534befb62c10498ccbd08f/reports/pyramidkv-metadata-net-cost-20261028/README.md)。
归属调整不构成新的运行、性能、质量或公开 wheel 验收。
