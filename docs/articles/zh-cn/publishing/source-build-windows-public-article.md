# Windows 源码构建：从 CMake 到可验证产物

TensorRtSharp4.0 的源码构建不是为了替代 NuGet 安装，而是给维护者一条能复核 native bridge、runtime 资产和 ABI guard 的本地路径。它适合在发布前确认 Windows 环境、CUDA/TensorRT SDK、CMake preset 和生成绑定是否仍然匹配。

## 适合

- 需要在 Windows 上从源码构建 TensorRtSharp4.0 的维护者。
- 需要排查 CUDA、TensorRT、cuDNN DLL 搜索路径的人。
- 需要理解 build-only、readonly diagnostics 与 package-consumer-runtime proof 边界的发布负责人。

## 准备环境

推荐先确认这些命令输出，并把 stdout/stderr 写入日志：

```powershell
dotnet --info
cmake --version
ninja --version
where cl
where nvcc
```

然后在源码目录运行生成和质量检查：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
```

native 构建使用发布候选 preset：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

这里的关键路径是 `native/CMakeLists.txt`、`native/src/tensorrt/v11/api.cpp`、`native/generated/bridge_api_catalog.g.h` 和 `native/generated/bridge_entrypoints.g.h`。如果这些文件发生变化，必须重新跑绑定生成和 quality gate。

## proof 边界

源码构建成功只能说明 build-only 路径健康。它不能证明用户能从公开包安装，也不能证明 post-publish 包可用，更不能替代 package-consumer-runtime proof。

| 产物 | 能说明什么 | 不能说明什么 |
|---|---|---|
| CMake configure/build 通过 | native toolchain 可构建 | 公开包可被外部 consumer 使用 |
| `dotnet build` 通过 | 托管项目可编译 | TensorRT runtime 已真实执行 |
| binding generator 通过 | manifest/source/generated 对齐 | deferred API 已有高层 wrapper |
| interface coverage matrix | 接口覆盖现状可审计 | release close 可通过 |

真正推动发布的仍然是 `artifacts/final-release/owner-external-proof-execution-result.input.json` 中 owner 回填的真实外部执行结果，并通过严格 validator。

## 配图建议

- 一张 Windows 终端截图，展示 CMake preset、`dotnet build` 和生成绑定连续通过。
- 一张目录截图，标出 `native/generated`、`artifacts/interface-coverage` 和 `artifacts/final-release` 的关系。
- 一张 evidence ladder 图，把 build-only 放在 proof 之前的低层级。

## 下一步

源码构建通过后，不要直接宣称发布完成。继续执行 clean external consumer，收集 stdout/stderr/merged transcript、SHA256、host metadata 和 owner review，再交给 `eng/Import-OwnerExternalProofExecutionResult.ps1` 与严格验证脚本处理。
