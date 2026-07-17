# TensorRT Native ABI Export Parity Candidate Review

日期：2026-07-18

## 结论

本批只修复“manifest 已实现、managed 已消费、但公共头未声明或 DLL 未导出”的 ABI parity 缺口，不新增高风险 public ownership surface。TRT8/TRT10/TRT11 最终 manifest-to-PE export 分别为 983/983、1078/1078、1226/1226，missing 0。

## 变更范围

| TensorRT line | 补齐头声明 | 最终 manifest entry | PE matched | Missing |
| --- | ---: | ---: | ---: | ---: |
| TRT8 | 17 | 983 | 983 | 0 |
| TRT10 | 31 | 1078 | 1078 | 0 |
| TRT11 | 14 | 1226 | 1226 | 0 |

TRT10 重建前为 1047/1078，精确缺 31 项。重建后的缺口归零证明新门禁检查的是实际 PE export，而不只是源码文本。

TRT11 的两个 refitter error-recorder snapshot entry 还缺少 source 编译接线，现已补齐 copied diagnostics、owner/index 校验、C++ exception 与 Windows SEH containment。返回值只包含 copied scalar、interface info 和 caller-buffer description，不返回 recorder pointer。

## 门禁语义

`eng/Test-TensorRtNativeAbiSurface.ps1`：

1. 解析每个版本的全部 manifest entry point。
2. 接受显式 `JYPPX_C_API(JYPPX_StatusCode)` 与既有 `*_DECL(entry)` 声明。
3. 使用完整 token 正则，不做入口名称子串匹配。
4. 传入 `-BridgePath` 时自动定位 `dumpbin` 并逐项验证 PE export。
5. 只生成 ABI surface evidence；不执行 inference、包发布、post-publish 晋级或 issue close。

## Runtime 边界

- TRT8/TRT10 的 registry、network、identity/enqueue smoke 已通过；TRT10 package runtime consumer 的 output match 为 True。
- TRT11 global/capability copied inventory 可用，但本机 CUDA driver 不兼容 CUDA 13.2，runtime/builder create 返回 null。该签名只被 Plugin Registry runner 作为 compatible-host skip；ABI 回归仍失败。
- callback trampoline、plugin create/register/deregister/load、allocator/resource acquire/release、device/borrowed pointer 和 ownership 不明确的 handle 继续 deferred。

## 发布边界

本记录不是 runtime proof、公开包 proof、post-publish proof 或 Owner 授权。Owner acceptance 仍为 0/9，禁止 NuGet push、GitHub Release upload 与自动关闭 issue。
