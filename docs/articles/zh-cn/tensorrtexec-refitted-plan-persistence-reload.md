# TensorRtExec Refitted Plan 持久化与独立 Reload

ONNX parser-refitter 把权重提交到内存中的 TensorRT engine，并不自动说明磁盘上的 stripped plan 已经包含
这些权重。要得到可部署的 refitted plan，需要把提交后的 engine 重新序列化，并证明新文件可以脱离原 engine
owner 独立加载和执行。

## 两个输出文件不能混用

```powershell
dotnet .\applications\TensorRtExec\bin\Debug\net8.0-windows\TensorRtExec.dll `
  --tensor-rt-line 10 `
  --onnx .\models\mnist.onnx `
  --saveEngine .\artifacts\mnist-stripped.plan `
  --stripWeights --refit `
  --refitFromOnnx .\models\mnist.onnx `
  --saveRefittedEngine .\artifacts\mnist-refitted.plan `
  --loadInputs Input3:.\inputs\digit-7-f32.bin `
  --iterations 1 --warmUp 0 --duration 0 `
  --exportOutput .\artifacts\mnist-refitted-output.json `
  --exportReport .\artifacts\mnist-refitted-run.json
```

`--saveEngine` 保存 builder 直接生成的 stripped plan；`--saveRefittedEngine` 保存 `RefitCudaEngine()`
提交后的 engine。两者必须是不同路径，refitted 输出也不能覆盖 ONNX build/refit source。当前覆盖策略允许替换
已有的 refitted 输出文件，但不会放宽上述源文件隔离规则。

这两个参数都是 managed 工具层契约。其中 `--saveEngine` 对应 trtexec 常见保存行为，
`--saveRefittedEngine` 是 TensorRtExec extension，不应写入 official trtexec option 清单。

## Owner-safe 执行顺序

持久化路径在任何 execution context 创建前完成以下步骤：

1. 从 stripped plan 创建第一个 `TensorRtEngine` owner。
2. 创建 owner 绑定的 `TensorRtRefitter` 与 `TensorRtOnnxParserRefitter`。
3. `RefitFromFile` 装载 ONNX 权重，复制 parser diagnostics。
4. `RefitCudaEngine()` 把权重提交到第一个 engine。
5. 创建 `TensorRtSerializationConfig`，显式清除并回读 `ExcludeWeights`，保证 refittable weights 进入输出。
6. `TensorRtEngine.Serialize(config)` 返回 bridge-owned `TensorRtHostMemory`。
7. `ToArray()` 把 plan 复制为 managed bytes，再写入 refitted 输出路径。
8. 释放第一个 refitted engine owner。
9. `TensorRtRuntime.DeserializeFromFile()` 从 refitted 文件创建第二个 engine owner。
10. 复制第二个 engine 的 refittable、I/O tensor、layer 和 profile metadata。
11. 只有 reload gate 完整通过，第二个 engine 才能创建 context 或进入 bounded runtime。

不能使用默认 `Serialize()` 完成这个步骤。TensorRT 10 的 stripped engine 默认序列化状态会继续排除可 refit
权重；这样的文件虽然能够反序列化并通过 I/O/layer/profile metadata 检查，实际输出却会变为零。完整权重 plan
在 TensorRT 10 reload 后可以不再保持 refittable，因此 `ReloadEngineRefittable` 是诊断字段，不是 context gate。

整个过程没有向 public API 暴露 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle` 或 borrowed serialization
pointer。host memory、原 engine 和 reload engine 都有清晰且互不混淆的 owner。

## `RefitPersistenceSnapshot`

JSON report 新增独立 snapshot，避免把 parser refit 与文件持久化混成一个状态：

- stripped plan path、length 与 SHA256；
- persisted plan path、length 与 SHA256；
- `SerializationFlagsBefore`、`SerializationFlagsAfter` 与 `RefittableWeightsIncludedInSerialization`；
- `ArtifactDiffersFromStrippedPlan`；
- `OriginalRefittedEngineDisposedBeforeReload`；
- `ReloadAttempted`、`ReloadSucceeded`、`ReloadEngineRefittable`；
- reload engine 的 I/O tensor、layer、optimization profile count；
- `ReloadContextCreationAllowed`；
- `ReloadEngineSelectedForRuntime` 与 `InferenceRanFromReloadedEngine`。

`--saveRefittedEngine` 只有在 serialize、文件差异、owner 释放、reload 与 metadata gate 全部成功时，才会
出现在 `AppliedOptions`。dry-run、TRT8 guard、dependency probe 或任一失败路径都只能进入 parse-only。

## 为什么还要第二进程

同一服务调用中释放原 engine 再 reload，能够证明对象 ownership 和文件反序列化边界，但进程仍共享同一套
runtime DLL 与环境。更强的持久化验证应启动第二个 TensorRtExec 进程，只提供 `--loadEngine`：

```powershell
dotnet .\applications\TensorRtExec\bin\Debug\net8.0-windows\TensorRtExec.dll `
  --tensor-rt-line 10 `
  --loadEngine .\artifacts\mnist-refitted.plan `
  --loadInputs Input3:.\inputs\digit-7-f32.bin `
  --iterations 1 --warmUp 0 --duration 0 `
  --exportOutput .\artifacts\mnist-second-process-output.json `
  --exportReport .\artifacts\mnist-second-process-run.json
```

第一进程 reload 输出、第二进程 load-engine 输出和 full-weight baseline 的 raw float SHA256 应完全一致。
第二进程不能引用 ONNX 文件，也不能复用第一进程的 engine wrapper；否则不能称为 independent process reload。

## 跨版本边界

- TensorRT 10：支持 parser-refitter、engine serialize 和 independent reload，可形成完整本地证据。
- TensorRT 11：代码路径相同，但必须在 runtime 可创建的兼容主机上重新取证；dependency probe 不能写成 applied。
- TensorRT 8：没有 ONNX parser-refitter API。dry-run 可以保留参数，non-dry 会在 native 调用前拒绝。

## 证据分类

本地持久化、same-process reload、second-process reload 和 baseline exact match 可以证明本机 source-tree
artifact 的 refitted-plan persistence。它们仍不是 NuGet/package-consumer runtime、跨机器兼容性、模型准确率、
公开发布或 release-close proof。clean package consumer 必须从声明的 package source restore，并独立记录
native assets、host metadata、命令、日志与 hash。

严格 evidence validator 位于：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-TrtexecRefittedPlanPersistenceEvidence.ps1 -Strict
```
