# Runtime Package 安装与排障深入

TensorRtSharp4.0 把 managed API、native bridge 和 NVIDIA runtime assets 分层打包。本文从用户安装角度解释如何选择 runtime package、如何配置本地依赖、如何判断错误来自包、驱动还是 proof 缺失。

## 适用读者

适合准备安装 NuGet/runtime 包的用户，也适合负责 Windows/Linux runtime package 发布矩阵和 clean consumer proof 的维护者。

## 解决问题

TensorRT、CUDA、cuDNN 版本组合很多。用户常把 restore 成功、native asset copied、dependency probe passed 和 runtime proof 混为一谈。本文解决安装路径、诊断顺序和 proof 边界问题。

## 背景与场景

Windows x64、Ubuntu 20.04/22.04/24.04、CUDA 12/13、TensorRT 8/10/11 都可能有不同包组合。安装教程必须说明 target RID、runtime key、driver requirement、local override 和 external clean consumer 的区别。

## 操作路径

1. 查 `pack/runtime/runtime-packages.manifest.json` 选择目标 runtime key。
2. 如果需要本地 vendor root，复制 `runtime-packages.local.example.json` 为 local override。
3. restore managed 包和 runtime split 包。
4. 运行 dependency probe，确认 native bridge 与 vendor DLL/SO 可加载。
5. 在兼容主机上执行 runtime smoke，并保存 host metadata、log、hash 和 validator。

## 代码与文件入口

- `pack/runtime/runtime-packages.manifest.json`
- `pack/runtime-split/split-runtime-packages.manifest.json`
- `eng/Collect-SplitRuntimeAssets.ps1`
- `eng/Test-PackageConsumer.ps1`
- `docs/articles/zh-cn/runtime-packages.md`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。安装成功不等于 public package proof，也不等于 post-publish verification。

## 下一步

下一步把常见安装错误拆成单独 FAQ：CUDA error 35、DLL/SO 加载失败、RID 不匹配、driver/runtime 不兼容和 runtime package key 选择错误。
