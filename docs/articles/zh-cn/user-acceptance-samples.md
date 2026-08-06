# 用户验收样例与 Smoke 目录

用户验收样例目录用于把“项目里有哪些可运行样例”和“哪些只是验证 runner”讲清楚。它不是运行结果汇总，也不会把当前机器未执行的 smoke 标记为 passed。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1
```

输出：

- `artifacts/user-acceptance/sample-smoke-catalog.json`
- `artifacts/user-acceptance/sample-smoke-catalog.md`
- `artifacts/user-acceptance/real-model-owner-handoff.json`
- `artifacts/user-acceptance/real-model-owner-handoff.md`

## 状态含义

- `ready-to-run`：仓库内已有可直接运行的用户样例，但仍需要匹配的 TensorRT/CUDA/cuDNN runtime。
- `asset-required`：样例代码存在，但需要用户提供模型、labels、图片等外部资产。
- `owner-action-required`：owner handoff、asset plan 或 runner evidence 仍在等待真实文件、hash、license 或日志。
- `cataloged-not-run`：项目存在并进入验收目录，但当前脚本没有执行它。
- `blocked-by-cuda-driver`：运行到 CUDA runtime 边界后被驱动兼容性阻塞。
- `not-run`：尚未纳入本轮执行或证据不足。

## 当前重点样例

低资产依赖、适合作为第一批文章和用户验收的样例：

- `samples/Performance/01.MultiStream`
- `samples/Inference/02.DynamicShapes`
- `samples/Inference/01.Bindings`
- `applications/OnnxToEngine`

需要外部模型资产的长教程：

- `samples/ComputerVision/01.Classification`
- `applications/YoloVision`

这些文章必须先说明模型来源、授权注意事项、输入 shape、labels、预处理和预期输出，不能只给一条运行命令。

`real-model-owner-handoff` 会把 Classification、YoloVision 和 YOLOX-S 的 owner 回填动作集中成一份清单。它不是运行结果，不会把 `asset-required` 改成通过；只有 sample run evidence record、manifest audit 和 user acceptance catalog 同时显示真实证据齐全后，文章才能写成 `real-model-runtime`。

## 证明边界

样例目录只能证明仓库中存在对应项目和运行入口。它不能证明：

- 当前机器具备兼容 GPU/driver/runtime。
- 所有 smoke 都已经通过。
- `IDebugListener::processDebugTensor` 已有真实 callback invocation。
- `IOutputAllocator` 或 allocator callback 已经完成真实 ownership 接管。
- Classification/YoloVision 的真实模型、labels、图片和日志已经回填。

真实 runtime proof 仍以 package consumer、smoke 输出和 final release dry run 为准。
