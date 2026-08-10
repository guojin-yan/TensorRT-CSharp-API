# TensorRtExec ONNX Stripped Plan Refit Lifecycle

`--stripWeights` 只证明 TensorRT 生成了剥离权重的 plan。要让这个 plan 进入推理，还需要一条完整、可审计且发生在 execution context 创建前的 refit 生命周期。

## 参数契约

```powershell
dotnet .\applications\OnnxToEngine\bin\Debug\net8.0\OnnxToEngine.dll `
  --tensor-rt-line 10 `
  --onnx .\models\model.onnx `
  --saveEngine .\artifacts\model-stripped.plan `
  --stripWeights `
  --refit `
  --refitFromOnnx .\models\model.onnx `
  --loadInputs input:.\inputs\input.fp32.bin `
  --iterations 1 --warmUp 0 --duration 0
```

`--refitFromOnnx` 是 managed 工具扩展，不是官方 trtexec 开关。它要求同时存在 `--onnx`、`--stripWeights` 和 `--refit`。当前不接受 `--loadEngine`，因为工具必须明确区分 build source、stripped plan 和权重来源，不能隐式猜测文件关系。

TensorRT 8 没有 ONNX parser-refitter API。dry-run 可以记录参数，但非 dry-run 会在 native 执行前拒绝。TensorRT 10/11 才进入实际生命周期。

## 执行顺序

1. builder config 设置并回读 `Refit` 与 `StripPlan`。
2. 生成并保存 stripped plan。
3. runtime 反序列化 stripped plan，确认 `ICudaEngine::isRefittable`。
4. 创建 owner 绑定的 `TensorRtRefitter` 和 `TensorRtOnnxParserRefitter`。
5. 通过 count/copy API 复制 missing/all named-weight inventory。
6. `RefitFromFile` 从显式 ONNX 文件装载权重，并复制 parser diagnostics。
7. `RefitCudaEngine` 把已装载权重提交到 engine。
8. 再次检查 missing inventory、parser error 和 engine refittable 状态。
9. 只有全部成功时，`ContextCreationAllowed=True`，后续才可以创建 context 和 enqueue。

这里不能省略第 7 步。实测 `RefitFromFile=True` 但未调用 `RefitCudaEngine` 时，MNIST enqueue 返回全零 logits；提交 engine 后，10 个 logits 的 40-byte SHA256 与 full-weight baseline 完全一致。

## 报告字段

JSON 报告中的 `RefitSnapshot` 包含：

- `EngineRefittableBefore/After`
- `ParserRefitReturned`
- `EngineRefitReturned`
- `MissingWeightsBefore/After`
- `AllWeightsBefore/After`
- `ParserErrorCount` 与 `CopiedDiagnosticCount`
- `ContextCreationAllowed`
- ONNX 文件大小与 SHA256

输出摘要额外记录原始 float 字节的 SHA256 和最多 64 个比较样本。hash 或样本只用于 baseline 比较，不会自动把 generic external runtime 提升为模型准确率证明。

## 已验证结果

TRT10.11 / CUDA12.9 的 TensorRT MNIST ONNX 实跑结果：

| 项目 | 结果 |
| --- | --- |
| parser refit | `True` |
| engine refit commit | `True` |
| all refittable weights | `6 -> 6` |
| missing weights | `0 -> 0` |
| parser errors | `0` |
| context gate | `True` |
| enqueue | 成功，output `[1,10]` |
| refit/baseline output | 40 bytes，SHA256 完全一致 |

TRT11.0 / CUDA12.9 也已在当前主机完成同一生命周期：stripped plan 反序列化、6 项 named-weight inventory、parser load、engine commit、`ExcludeWeights` 清除/readback、原 owner 释放、full-weight plan reload、context gate、enqueue 与 10 值零 mismatch reference comparison 均通过。随后启动的第二个独立进程只使用 `--loadEngine`，不引用 ONNX 或 refit 参数，仍得到相同输出 SHA256。两份报告都通过 `eng/Test-TensorRtExecReport.ps1 -Strict` 的 `69/69` 项检查。

TRT11 证据根目录为 `artifacts/real-case/tensorrtexec-trt11-refit-lifecycle-20260810-091859/`。仓库内固定 reference 来自同一 MNIST 输入的未审核 runtime 输出，所以这里证明的是本机 source-tree refit/reload 与数值回归，不是模型准确率、package consumer、public feed、post-publish 或 release proof。2026-07-22 的 `dependency-probe-only` 报告继续作为历史环境记录保留，不代表当前 TRT11 refit 状态。

## 证据边界

只指定 `--refitFromOnnx` 时，成功 refit 的对象只存在于当前进程内，`--saveEngine` 指向的文件仍是 stripped plan。需要 durable artifact 时必须另行指定 `--saveRefittedEngine`，并按“Refitted Plan 持久化与独立 Reload”文章验证新文件。本节原有结果本身不是 persistence、模型准确率、公开包消费或发布证明。完整 compact evidence 位于 `artifacts/interface-coverage/trtexec-onnx-refit-lifecycle-evidence.json`，严格验证命令为：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-TrtexecOnnxRefitLifecycleEvidence.ps1 -Strict
```
