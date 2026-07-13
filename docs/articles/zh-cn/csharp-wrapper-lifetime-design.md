# C# Wrapper Lifetime 设计

## 适用读者

本文面向维护 TensorRtSharp 高层 C# wrapper 的开发者，重点关注 native handle、borrowed pointer、owned resource、Dispose、SafeHandle 和跨 ABI 错误边界。

## 解决问题

C# wrapper 的难点不是简单 P/Invoke，而是生命周期。本文说明 public API 应隐藏裸指针，明确 owned 与 borrowed 的差异，使用复制字符串、buffer 输出和不可变快照降低风险，同时强调 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不是 runtime proof。

## 背景与场景

TensorRT API 中有 builder、runtime、engine、context、plugin creator、error recorder、allocator、debug listener 等多种对象。部分对象由 TensorRT 拥有，部分需要调用方释放，部分只能在回调期间短暂借用。C# 层必须把这些边界表达清楚，否则很容易出现悬空指针、重复释放或跨 ABI 异常。

## 实现路径

1. 对 owned native object 使用明确的释放路径，并在 C# 层封装为 disposable 类型。
2. 对 borrowed pointer 不暴露 `IntPtr`，优先复制为 string、record、array 或 snapshot。
3. 对 caller buffer 输出提供长度查询和 copy 两段式 API。
4. 对 callback/allocator 先设计 owner ledger 和 nothrow trampoline，再进入 runtime proof。
5. 对跨版本差异使用 version guard，避免 TRT8/TRT10/TRT11 混用语义。

## 代码与文件入口

- `src/JYPPX.TensorRtSharp`：高层 C# wrapper。
- `native/src/tensorrt/common`：跨版本 ABI adapter。
- `docs/articles/zh-cn/plugin-ownership-boundary.md`：plugin ownership 边界。
- `docs/articles/zh-cn/callback-allocator-boundary-guide.md`：callback/allocator 边界。
- `docs/articles/zh-cn/deferred-api-real-completion-review.md`：真实完成度复审。

## 图示建议

建议用三色表展示 owned、borrowed 和 snapshot：owned 有释放责任，borrowed 只能在 native 作用域内使用，snapshot 是 C# public API 推荐输出。

## 边界说明

Wrapper lifetime 设计是发布硬化的一部分，但不是 proof 本身。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report 和 readonly diagnostics 仍需被严格标注为非 runtime proof。

## 下一步

下一轮应梳理 public API 中所有 handle 暴露点，把可替换的 borrowed pointer 输出改成 record/snapshot，并为无法替换的内部 handle 增加文档和测试约束。
