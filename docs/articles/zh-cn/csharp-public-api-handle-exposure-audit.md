# C# Public API Handle 暴露审计

## 适用读者

本文面向维护 TensorRtSharp public API 的开发者，重点说明如何审计 public surface 中的 `IntPtr`、`nint`、native handle 和 borrowed pointer 暴露。

## 解决问题

高层 C# wrapper 的目标不是把所有 native 指针直接暴露给用户，而是用 owned resource、snapshot、record、string 和 array 表达安全边界。本文给出 public API 审计路径，同时强调 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不是 runtime proof。

本阶段质量门禁明确要求高层 public API 不暴露裸 `IntPtr`、`nint` 或其他 borrowed pointer，尤其不能暴露 plugin creator、field collection、debug tensor、output buffer 或 device pointer 这类 native 生命周期不归 C# 调用方管理的对象。

## 背景与场景

TensorRT 中很多对象由 native runtime 拥有，例如 plugin creator、field collection、debug tensor 和 callback 上下文。如果 public API 暴露裸 `IntPtr`，用户可能在对象生命周期结束后继续使用，导致悬空指针或重复释放。审计测试应允许内部 interop 使用指针，但禁止高层公开类型泄漏 borrowed pointer。

## 实现路径

1. 扫描 `src/JYPPX.TensorRtSharp` public 类型和 public 成员中的 `IntPtr`、`UIntPtr`、`nint`、`nuint`。
2. 将允许的低层 interop、SafeHandle 或明确 owner 类型列入白名单。
3. 对 plugin creator、field metadata、debug listener borrowed tensor、output allocator buffer 等高风险词使用更严格断言。
4. 将可替换的 pointer 输出改为 string、record、array 或 immutable snapshot。
5. 在质量测试中锁定禁止词，避免后续回归。

## 代码与文件入口

- `src/JYPPX.TensorRtSharp`：public API 主体。
- `src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs`：pointer-free plugin inventory 示例。
- `docs/articles/zh-cn/csharp-wrapper-lifetime-design.md`：生命周期设计。
- `docs/articles/zh-cn/plugin-ownership-boundary.md`：plugin ownership 边界。
- `tests/JYPPX.ProjectQuality.Tests`：public API 审计测试入口。

## 图示建议

建议用三层图：native pointer、internal interop、public snapshot。只有 snapshot 层暴露给用户，native pointer 停留在内部。

## 边界说明

Public API handle audit 提升 API 安全性，但不是 runtime proof。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 仍然不能替代真实运行证据。

## 下一步

下一轮应把审计测试扩展为 Roslyn 或 reflection 级别，区分 internal interop 与 public API，并对已知例外写清楚 owner 语义。
