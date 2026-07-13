# Deferred API 真实完成度复盘

TensorRtSharp4.0 的 deferred API 主线已经从“missing 接口清零”切换到“deferred 边界提升”。本文解释为什么 manifest/source 匹配不等于真实可用，以及如何判断某个接口已经从占位入口推进为可调用、可诊断、可测试的真实 API。

## 适用读者

适合维护 TensorRT/CUDA native bridge、C# wrapper 和 release quality gate 的开发者，也适合想理解项目真实完成度的用户。

## 解决问题

接口占位容易制造“100% 完成”的错觉。本文解决真实完成度口径：非 deferred native 实现、高层 C# wrapper、version guard、smoke/package consumer 验证、文档和 proof 边界必须同时看。

## 背景与场景

当前优先提升只读、查询型、部署关键型 API，例如 plugin registry inventory、builder config getter、engine metadata、parser diagnostics。callback、裸指针、外部资源和跨语言 ownership API 必须谨慎推进。

## 操作路径

1. 从 `artifacts/interface-coverage/tensorrt-interface-comparison.csv` 筛 deferred API。
2. 按 ownership 风险分层：readonly、borrowed pointer、callback、external resource。
3. 对 readonly API 补 manifest 参数、native source、C# interop、高层 wrapper 和 XML 注释。
4. 运行 binding generator、quality tests、smoke 或 package consumer check。
5. 在 release docs 中记录仍未完成的 runtime proof 和 owner blocker。

## 代码与文件入口

- `native/manifests/tensorrt/v8`
- `native/manifests/tensorrt/v10`
- `native/manifests/tensorrt/v11`
- `native/src/tensorrt/common`
- `src/JYPPX.TensorRtSharp`
- `artifacts/interface-coverage/project-completion-review.md`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。readonly API 完成也不能证明 plugin create/enqueue、callback trampoline 或外部 resource ownership 已完成。

## 下一步

下一步继续按高用户价值、低 ownership 风险推进 deferred API：优先补只读查询和 metadata copy，避免 public API 暴露裸 `IntPtr` 或 borrowed pointer。
