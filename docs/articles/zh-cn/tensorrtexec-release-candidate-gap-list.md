# TensorRtExec 发布候选缺口清单

TensorRtExec 的方向是复刻官方 `trtexec` 的模型转换体验，同时提供 C# CLI 和 WinForms 页面。当前它已经覆盖了不少 build/report 能力，但离“可以作为发布候选功能集合”还差一层工作清单：哪些能力只是参数解析，哪些能生成报告，哪些仍然 deferred，哪些完全不能作为 runtime proof。

仓库中的机器可读清单位于：

```text
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json
```

配套 Markdown 位于：

```text
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.md
```

## 怎么读这个清单

每一项都包含：

- 官方 `trtexec` 对应参数。
- 当前状态：`implemented`、`partial`、`implemented-report`、`diagnostic`、`deferred`、`owner-action-required`。
- 是否支持 CLI。
- 是否支持 WinForms。
- 是否能作为 runtime proof。
- 是否能作为 package-consumer-runtime proof。
- 下一步实现路径。

## 当前最重要的缺口

第一类是安全反序列化和 binding diagnostics。`--loadEngine` 目前不能被宣传成真实运行证明；它需要先有 read-only deserialize、binding metadata、error diagnostics，再考虑 enqueue。

第二类是 builder config readback。`--workspace` 与已知 `--memPoolSize` pool 现在会在真实 build 中调用 typed setter 并用 getter read back；dynamic profile 等仍需要报告和真实模型证据。readback 只说明 TensorRT 接收了 builder 配置，不是 runtime 输出或 package-consumer proof。

第三类是 WinForms parity。GUI 不应该只是“能打开页面”，而是要能覆盖 CLI 的主要参数、生成可复制命令、展示 report 摘要和错误诊断。

第四类是 proof 边界。TensorRtExec 可以辅助生成 build report 和 sidecar，但不能替代 YoloVision real-model-runtime proof，更不能替代 clean external consumer 的 package-consumer-runtime proof。

## 配图建议

- 一张 CLI 参数到 WinForms 控件的映射图。
- 一张 gap list 表格截图。
- 一张 evidence ladder 图，说明 build report、sample run、real-model-runtime、package-consumer-runtime 的区别。

## 下一步

下一阶段优先把 `load-engine`、`binding-metadata` 和 `winforms-command-surface` 三项拆成实现任务；`workspace-memory-pool` 转入 compatible-host owner build record。所有实现都需要同步更新 CLI、WinForms、文档、测试和 proof 边界说明。
