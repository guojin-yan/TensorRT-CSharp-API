# Runtime Package 最小 Smoke 命令模板

## 适用读者

本文面向需要验证 TensorRtSharp runtime package 是否能在 Windows/Linux clean consumer 中加载的维护者和外部试用用户。

## 解决问题

安装成功不等于 native runtime 可加载。本文给出最小 smoke 命令模板，帮助用户记录 restore、build、native load、engine deserialize 或 sample run 的证据，同时明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不能替代 runtime proof。

## 背景与场景

Runtime package proof 必须来自干净 consumer：不能使用仓库内 ProjectReference，不能直接引用本地 `.nupkg` 文件作为最终公开 proof，不能只依赖 local feed。最小 smoke 的价值是先证明 native library resolution 和基础 TensorRT/CUDA 调用可执行，再进入更完整样例运行。

## 操作路径

1. 在仓库外创建 clean consumer 项目，并从目标 package source 安装主包和 runtime package。
2. 执行 restore/build，记录 package source、package version、RID 和 lock file。
3. 运行最小 native load 程序，输出 CUDA、TensorRT、cuDNN library load 状态。
4. 如果有 engine，执行 deserialize smoke；如果有模型资产，执行 YoloVision sample run。
5. 保存 stdout、stderr、exit code、host metadata、package hash 和 validator 结果。

## 代码与文件入口

- `docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`：package consumer proof playbook。
- `docs/articles/zh-cn/runtime-package-windows-linux-install-faq.md`：安装 FAQ。
- `pack/runtime`：runtime package 定义。
- `tests/JYPPX.ProjectQuality.Tests`：proof 与 schema 质量测试。
- `artifacts/stable-runtime-package-source`：本地稳定源仅可用于预检，不可替代公开 proof。

## 图示建议

建议画一张命令阶梯：restore -> build -> native load -> deserialize -> sample run -> validator。每级标注是否只是 precheck，是否可进入 proof candidate。

## 边界说明

最小 smoke 命令模板仍是 template；只有被真实执行并通过 validator 的外部 clean consumer 记录才可能成为 runtime proof。build-only、dry-run、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report 和 readonly diagnostics 不能替代该记录。

## 下一步

下一轮应提供 Windows PowerShell 与 Linux bash 两套命令草稿，并将输出字段与 owner proof input schema 对齐。
