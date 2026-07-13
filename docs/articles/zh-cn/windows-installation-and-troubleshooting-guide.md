# Windows 安装与排查：CUDA、TensorRT、PATH 与 Runtime Package

## 适用读者

这篇文章适合在 Windows 上安装、试用或排查 TensorRtSharp 4.0 的用户，包括桌面应用开发者、WinForms 工具使用者、CI/self-hosted runner 维护者。

## 解决问题

Windows 上常见问题不是 C# 编译失败，而是 native load、driver/runtime 不匹配、PATH 中混入旧版本 DLL、runtime package 与本机 TensorRT/CUDA/cuDNN 不一致。TensorRtSharp 4.0 的做法是尽量通过 runtime package 和诊断 artifact 降低环境不确定性，但最终仍需要用户确认 driver、CUDA、TensorRT 和 cuDNN 组合。

## 安装前检查

建议先执行：

```powershell
dotnet --info
nvidia-smi
Get-Command nvcc -ErrorAction SilentlyContinue
```

如果 `nvidia-smi` 不可用，先处理 NVIDIA driver。若 driver 支持的 CUDA runtime 低于 runtime package 需求，可能出现 CUDA error 35 或 native 初始化失败。相关排查可继续看 `docs/articles/zh-cn/cuda-error-35-troubleshooting.md`。

## 安装和运行建议

普通用户先从 sample help 命令开始，不要直接拿生产模型做首测：

```powershell
dotnet run --project .\applications\TensorRtExec -- --help
dotnet run --project .\samples\OnnxToEngine -- --help
dotnet run --project .\samples\YoloVision -- --help
```

如果使用自己的 consumer 项目，请安装 managed 包和匹配 runtime package。确认 `bin\Release\net8.0` 或发布目录中 native assets 已被复制。若出现 DLL load 失败，先检查 runtime key、RID、PATH、当前工作目录和是否混入旧 TensorRT DLL。

## 常见故障路径

第一类是 driver 不支持 runtime：表现为初始化失败或 CUDA driver/runtime mismatch。第二类是 native DLL 搜索路径错误：表现为找不到 TensorRT/CUDA/cuDNN 依赖。第三类是 package 与 host 不匹配：表现为 build 正常但运行失败。第四类是把 build/report 当成 proof：这会让 release 判断提前失真。

## 边界说明

Windows 安装成功、sample help 成功、TensorRtExec GUI 截图、build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不是 runtime proof。release proof 仍需要仓库外 clean consumer、public package source、真实 smoke 日志、hash、host/package metadata 和 strict validator。

## 下一步

如果你要在 Windows 上继续验证真实模型，请先选择 `samples/OnnxToEngine` 或 `samples/YoloVision`，准备模型来源、license、labels、input shape 和 SHA256。若你的目标是 release proof，请转到 `package-consumer-runtime-proof-clean-consumer-guide.md`。
