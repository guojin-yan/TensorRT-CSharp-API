# Linux 安装与 Runner 边界：容器、Native 包与 Proof 的距离

## 适用读者

这篇文章适合 Linux 部署工程师、CI runner 维护者、准备在 Ubuntu 或容器中运行 TensorRtSharp 4.0 的团队。

## 解决问题

Linux 上的难点通常在 runner：容器里能 restore 不代表宿主机 driver 可用，CI 中能 build 不代表 TensorRT runtime 可加载，远程 runner 输出也不能自动变成 package-consumer-runtime proof。项目需要把 Linux runner evidence、runtime smoke、public package source 和 strict validator 分开看。

## 安装前检查

Linux 环境建议先确认：

```bash
dotnet --info
nvidia-smi
ldconfig -p | grep -E "nvinfer|cudart|cudnn"
```

如果使用容器，确认 NVIDIA Container Toolkit、driver passthrough、CUDA compatibility package 和 TensorRT shared library 路径。不要只依赖 Dockerfile 或 workflow yaml 判断运行能力；最终 runtime 行为必须由真实 host 日志证明。

## Runner 与 Runtime Package

Linux runtime package 应与 RID、CUDA、TensorRT、cuDNN 组合一致。对于 self-hosted runner，建议把 OS version、kernel、GPU、driver、CUDA runtime、TensorRT version、cuDNN version 写进 proof input。对于 GitHub hosted runner，如果没有 GPU/TensorRT 环境，它只能做 build/preflight/documentation 类检查，不能成为 runtime proof。

## 操作建议

```bash
dotnet restore
dotnet build -c Release --no-restore
dotnet run -c Release --no-build --project samples/OnnxToEngine -- --help
```

这些命令适合确认基础工具链。真实 smoke 需要模型资产和 native runtime。release 级别 proof 还要求仓库外 clean consumer、public package source、restore/build/smoke 日志与 hash。

## 边界说明

Linux runner build、build-only、容器启动、dry-run、template、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不是 runtime proof。Linux proof 必须来自真实兼容 host 上的执行日志、hash、host metadata 和 strict validator。

## 下一步

如果只是使用项目，下一步选择 runtime package 并运行 sample。若你是 release owner，请用 `artifacts/final-release/clean-consumer-proof-owner-execution-pack.md` 作为执行清单，并在真实 public package source 出现后回填 owner input。
