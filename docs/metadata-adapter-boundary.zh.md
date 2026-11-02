# 独立适配层的归属

[English](metadata-adapter-boundary.md) | 简体中文

PR26 的旧方案等待共享元数据接口。0.3 按本次移除安装、导入和运行时依赖的要求，
直接在 PyramidKV 仓库维护完整适配层；详见[依赖审计与设计](superpowers/specs/2026-11-02-standalone-adapter-design.md)
和[已实现的宿主契约](host-contract.md)。

调度事务、缓存绑定、查询观察、逐层元数据、修正张量缓存、持久槽位、MTP 修正、
重置和原生回退现在均由 `vllm_ascend_pyramidkv.adapter` 管理。算法使用本地类型和
直接工厂，不再通过外部方法注册表构建共享 provider。默认关闭，启用时检查宿主接口；
与其他已启用的 KV 修改适配层冲突时明确报错。

[共享 PR33](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33) 保持历史源码参考，
本次不修改上游 PR，也不宣称已设计通用上游接口。[旧实验规程](benchmark/README.zh.md)
保留原先归属，旧测量不能作为独立适配层的新证据。发布前遵循[独立验证规程](standalone-validation.md)。
