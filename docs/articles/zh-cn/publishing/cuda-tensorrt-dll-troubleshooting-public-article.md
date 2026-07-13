# CUDA / TensorRT DLL 加载问题排查

在 Windows 上使用 TensorRT C# binding，最常见的问题不是 C# 语法，而是 native 依赖加载：CUDA driver、CUDA runtime、TensorRT DLL、cuDNN、Visual C++ runtime、PATH、进程位数和 runtime package 都必须匹配。

本文给出面向用户的排查路径，帮助把“DLL 找不到”变成可定位、可复现、可反馈的问题。

## 适合谁阅读

- 遇到 CUDA/TensorRT DLL 加载失败的新用户。
- 需要维护 runtime package、native assets copied 和 dependency probe 的发布负责人。
- 准备采集 clean external consumer proof 的 owner。

## 先确认版本组合

排查前记录：

- GPU 型号。
- NVIDIA driver 版本。
- CUDA runtime 版本。
- TensorRT 主版本。
- cuDNN 版本。
- 目标 RID，例如 `win-x64`。
- 使用的 TensorRtSharp managed package 与 runtime package。

这些字段后续也会进入 owner proof 输入，不能只靠截图或口头描述。

## 常见错误

### 找不到 CUDA/TensorRT DLL

通常是 runtime assets 没复制到输出目录，或 PATH 中没有对应 TensorRT/CUDA bin 目录。先确认输出目录是否包含 native bridge 和目标 runtime DLL，再确认进程启动时 PATH。

### 版本不匹配

TensorRT 8、10、11 的 ABI 和 API 都有差异。项目通过 version guard 和 runtime package key 区分主线，不要把不同主版本 DLL 混在同一输出目录。

### x86/x64 不匹配

TensorRT/CUDA 基本是 x64 路径。确保 .NET 进程、native DLL、runtime package 都是 x64。

### 本地可跑但外部 consumer 不可跑

这类问题最容易被 local feed 或 ProjectReference 掩盖。真实 package-consumer-runtime proof 必须使用公开包和干净外部项目。

## 推荐检查命令

```powershell
dotnet --info
nvidia-smi
Get-ChildItem .\bin\Debug\net8.0 -Filter *.dll
$env:PATH -split ';'
```

如果使用样例，优先跑 dependency probe 或 smoke runner，而不是直接进入复杂模型。

## 不能作为 proof 的材料

- “我机器上能 build”。
- 本地 `.nupkg`。
- local feed。
- ProjectReference。
- 只包含 build-only 的 TensorRtExec report。
- 没有 hash 的日志片段。

## 配图建议

- DLL 搜索路径示意图。
- managed package / runtime package / system CUDA DLL 的分层图。
- 常见错误信息到排查动作的表格。

## 下一步

后续应把每个 runtime package key 的依赖 DLL 列成机器可读清单，并让 clean consumer validator 检查 native assets copied、dependency probe status 和 smoke status。
