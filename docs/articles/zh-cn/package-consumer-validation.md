# 包消费端验证

`eng/Test-PackageConsumer.ps1` 用于验证托管主包与 runtime 包在真实消费端项目中的还原、构建、native asset 复制和可选 smoke。

当前 Windows 重点 runtime key：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

CUDA `12.9` 已安装。目标为 CUDA `12.9` 的包必须使用 CUDA `12.9` 以及匹配 TensorRT/cuDNN 资产完成验证；之前 CUDA `12.3` 的临时 fallback 已废弃。

示例命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 -RunSmoke -SmokeRuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9
```

2026-06-12 本机最新证据：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：`16/16` native assets，smoke `passed`，探针输出 TensorRT `10.11.0`、CUDA `11.8`。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：`19/19` native asset patterns，smoke `passed`，探针输出 TensorRT `10.11.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：`19/19` native asset patterns，smoke `passed`，探针输出 TensorRT `11.0.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：`19/19` native asset patterns，restore/build/native-copy 通过；当前 driver/runtime 栈不请求 smoke。
- managed package：`JYPPX.TensorRT.CSharp.API 4.0.0-alpha.1`
- 报告：`artifacts/package-consumer/package-consumer-validation-summary.md`

如果 Windows Defender Application Control / 应用控制策略阻止新构建的消费端输出，并出现 `0x800711C7`，可以显式传入 `-SignConsumerOutput`。该开关会在 smoke 前使用本地开发代码签名证书签名生成的消费端程序、托管程序集和桥接 DLL：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 -RunSmoke -SmokeRuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 -SignConsumerOutput
```

消费端验证应至少检查：

- managed package restore
- runtime package restore
- native 资产复制数量
- 缺失 native 资产列表
- 可选 smoke 结果

`win-x64-trt11.0-cuda13.2-cudnn9.22` 不能仅凭 restore/build/native-copy 通过就视为发布可用；在 CUDA 13-capable driver/runtime 上通过 runtime/builder smoke 前，readiness 必须保持 blocked。

正式发布前仍需完成 NVIDIA CUDA / cuDNN / TensorRT 再分发许可复核。
