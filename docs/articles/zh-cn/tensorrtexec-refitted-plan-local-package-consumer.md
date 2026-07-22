# TensorRtExec Refitted Plan 本地包 Consumer 实战

上一阶段证明了 `--saveRefittedEngine` 可以把 ONNX stripped plan 的 refit 结果持久化，并在同进程、第二进程和
full-weight baseline 三条路径得到相同输出。那仍然是源码树中的 TensorRtExec 运行。本篇继续把边界推进到一个
完全独立的 `PackageReference` consumer：它从声明的本地 NuGet source 还原 managed 与 bridge 包，不引用源码项目，
也不从源码目录手工加载程序集。

## 这次要证明什么

完整链路必须同时满足：

1. consumer 工作区位于源码仓库之外，并使用自己的 `RestorePackagesPath`；
2. `NuGet.config` 只有 managed 与 TRT10 bridge 两个本地 source，不启用 nuget.org；
3. 项目只有两个 `PackageReference`，没有 `ProjectReference`；
4. persisted plan 和 float input 被复制到 consumer 工作区，执行命令不直接读取源码目录中的资产；
5. `JYPPX_NATIVE_BRIDGE_PATH` 与 development probing 均被清空，bridge 必须来自包的 runtime asset；
6. consumer 独立创建 runtime、engine、context、bindings 和 CUDA stream owner；
7. enqueue、readback 和 owner scope 全部完成，raw float 输出 SHA256 与既有 baseline 精确一致；
8. 采集 package/native inventory、host、命令和日志 hash 后删除 consumer 工作区。

它证明的是 `local-package-consumer-refitted-plan-runtime`，不是公开 feed 下载、post-publish、模型准确率或发布授权。

## 前置资产

当前证据使用 TensorRT 10.11 / CUDA 12.9 bridge，以及 TensorRT MNIST sample 的 digit-7 输入。必须先存在：

```text
artifacts/managed/JYPPX.TensorRT.CSharp.API.4.0.0.nupkg
artifacts/runtime-split-nupkg/win-x64-trt10.11-cuda12.9-cudnn9.22/*.nupkg
artifacts/real-case/trtexec-refitted-plan-persistence/mnist-refitted-persisted.plan
artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-input-f32.bin
```

persisted plan 必须来自显式 serialization config：先清除并回读 `ExcludeWeights`，再调用
`Serialize(serializationConfig)`。不能用 stripped engine 的默认 `Serialize()` 代替，否则 plan 虽能 reload，权重仍
可能被排除并产生全零输出。

本阶段固定输入与输出基线为：

```text
plan SHA256  = 5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb
input SHA256 = 81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564
output SHA256= 6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041
```

## Consumer 项目

模板位于 `samples/RefittedPlan.PackageConsumer`。运行器复制模板后注入实际包 ID、版本、RID 与隔离 cache 路径：

```xml
<PackageReference Include="JYPPX.TensorRT.CSharp.API" Version="4.0.0" />
<PackageReference Include="JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge"
                  Version="4.0.0" />
```

程序只使用公开高层 wrapper：

```csharp
using TensorRtLogger logger = new TensorRtLogger(TensorRtApiLine.TensorRt10);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtEngine engine = runtime.DeserializeFromFile(planPath);
using TensorRtExecutionContext context = engine.CreateExecutionContext();
using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex: 0);
using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
```

随后复制 `Input3` 的 784 个 float，分配 `Plus214_Output_0` 的 10 个 float，完成 bind、readiness、enqueue 和
readback。只有 `RunPersistedPlan` 返回并离开全部 `using` scope 后，程序才输出 `OwnerScopeExited=True`。

## 一条命令执行

从仓库根目录运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-TrtexecRefittedPlanPackageConsumer.ps1 -Strict
```

默认工作区是外层 E 盘的 `.package-consumer-work`。脚本会依次完成：

- 校验源 plan/input SHA；
- 创建只有两个本地 source 的 `NuGet.config`；
- `dotnet restore --force --no-cache`；
- Release build；
- 检查 `project.assets.json` 的包版本和隔离 cache；
- 检查 managed assembly 与 `jyppxtrtbridge.dll` 都位于 consumer output；
- 设置系统 TensorRT/CUDA runtime 搜索路径，但清除源码开发探测；
- 执行 copied plan/input；
- 采集日志、包、bridge、`nvinfer_10.dll`、`cudart64_12.dll` 和 host hash；
- 删除 consumer 工作区；
- 运行严格 compact evidence validator。

成功输出应包含：

```text
ProjectReference=False
EngineRefittable=False
EngineIOTensorCount=2
EngineLayerCount=5
EngineOptimizationProfileCount=1
BindingsReadyForEnqueue=True
EnqueueCompleted=True
OutputExactMatch=True
PredictedIndex=7
OwnerScopeExited=True
PackageConsumerRuntime=Passed
```

完整权重 TRT10 engine reload 后 `EngineRefittable=False` 是已验证事实，不是失败条件。runtime gate 使用可执行
metadata、binding readiness、enqueue 和 output exact match，不重复要求 full-weight plan 仍可 refit。

## 证据文件

本机 raw 记录位于 ignored 的：

```text
artifacts/package-consumer/trtexec-refitted-plan/<runtime-key>/
```

其中包含实际路径、命令、stdout/stderr 和 native asset 路径。可提交的 path-free 证据位于：

```text
artifacts/interface-coverage/trtexec-refitted-plan-package-consumer-evidence.json
artifacts/interface-coverage/trtexec-refitted-plan-package-consumer-validation.json
```

严格校验器会把 consumer plan/output SHA 与上一阶段
`trtexec-refitted-plan-persistence-evidence.json` 交叉比对，因此不能通过手工改一份孤立 JSON 来伪造通过。

## 常见失败

- `Persisted plan SHA256 mismatch`：输入不是上一阶段完整权重 plan，或文件已被改写。
- `Restore did not resolve both target packages`：本地 feed 缺包、版本不一致，或 restore 没有进入隔离 cache。
- `Bridge asset was not copied`：bridge-only nupkg 的 RID/asset 布局不正确。
- `Bindings are not ready`：plan I/O、shape 或 buffer binding 不满足 enqueue 条件。
- `Output SHA256 mismatch`：权重、输入、runtime 版本或执行结果与已验证 baseline 不一致；不得降级为成功。
- `workspaceRemovedAfterValidation=False`：隔离目录未清理，证据保持失败，先完成精确清理再重跑。

## 证据边界

这条链路证明本地构建的 managed/bridge nupkg 能在一个无源码引用的 consumer 中执行 persisted refitted plan。
它不证明包已从 NuGet.org 或 GitHub Packages 下载，不证明另一台机器或另一 TensorRT/CUDA 组合兼容，也不授权
NuGet push、GitHub Release upload、issue close 或其它公开发布动作。
