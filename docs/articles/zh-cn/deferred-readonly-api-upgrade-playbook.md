# Deferred Readonly API 升级 Playbook

## 适用读者

本文面向继续推进 deferred 边界提升的维护者，尤其是需要从 manifest/source 100% 匹配转向真实可用 API、wrapper 和 smoke proof 的开发者。

## 解决问题

接口对齐完成不代表 API 可用。本文给出 deferred readonly API 的升级顺序，帮助维护者选择低 ownership 风险的只读 API，补 native/source、C# interop、高层 wrapper、XML 注释和 smoke 测试，而不是删除 deferred 记录制造完成度。

## 背景与场景

TensorRT 和 CUDA 中很多接口涉及 borrowed pointer、callback、allocator、plugin instance、driver entrypoint 和跨 ABI 生命周期。项目当前策略是先提升安全只读面，再逐步处理 owner 明确的资源桥接。Plugin registry inventory、engine inspector、builder config readback 和 error recorder snapshot 都属于较适合先推进的区域。

## 实现路径

1. 用 coverage CSV 和 `rg` 找出目标关键词相关 deferred 行。
2. 每批选择 5 到 15 个无 ownership 接管、无 callback、无裸 borrowed pointer 暴露的 API。
3. 将 no-arg deferred 改为真实参数、返回码和 caller buffer/copy 输出模式。
4. 更新 C# interop、高层 wrapper、XML 注释和错误码映射。
5. 增加 smoke 或 quality test，确认非 deferred 实现可被实际调用。

## 代码与文件入口

- `artifacts/interface-coverage/tensorrt-interface-comparison.csv`：接口覆盖矩阵。
- `native/manifests/tensorrt`：manifest 声明。
- `native/src/tensorrt`：native source 实现。
- `src/JYPPX.TensorRtSharp`：C# interop 与高层 wrapper。
- `tests/JYPPX.ProjectQuality.Tests`：质量门禁。

## 图示建议

建议配一张升级漏斗图：`deferred row -> safe readonly candidate -> native adapter -> C# wrapper -> smoke -> release readiness`，并在漏斗外列出 callback、allocator、plugin instance 等暂缓项。

## 边界说明

Deferred readonly API 升级可以提高真实完成度，但不能把 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report 或 readonly diagnostics 直接变成 runtime proof。API smoke 与 package consumer runtime proof 仍是两条不同证据链。

## 下一步

下一轮应基于当前 coverage matrix 选择下一个只读批次，优先覆盖 engine/layer metadata、builder config readback 或 plugin field metadata，并在每批完成后更新开发日记和下一阶段提示词。
