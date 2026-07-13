# Package Readiness Summary 怎么读

`artifacts/package-readiness/runtime-package-readiness-summary.md` 是判断 runtime package 当前状态的主证据之一。它不是简单的“通过/失败”表，而是把 managed package、bridge package、split components、full runtime package、consumer report、vendor root 和 runtime smoke 分开表达。

读这份报告时，最重要的是区分 package completeness、consumer build evidence 和 runtime execution evidence。

## 总览表

报告顶部类似：

```text
| Runtime key | Managed | Bridge package | Bridge consumer | Split components | Split collection | Split collection consumer | Full vendor inputs | Full runtime package | Full consumer | Overall |
| win-x64-trt11.0-cuda13.2-cudnn9.22 | ready | ready | ready | ready 3/3 | ready | ready | ready | ready | ready | ready |
```

这说明当前 runtime key 的包完整性、消费端报告和 vendor input 检查都达到了 readiness 要求。`Overall=ready` 是 package/readiness 层面的 ready，不是所有 runtime API 都已经真实执行通过。

## Split 与 full package

`split components: ready 3/3` 表示以下组件均存在：

- `Bridge`
- `CudaCudnn`
- `TensorRt`

`split collection package: ready` 表示轻量 collection 包也存在，可以固定这组组件版本。

`full runtime package: ready` 表示完整 runtime nupkg 存在，适合 full package consumer validation。

## Consumer 状态

consumer 分为三类：

- bridge consumer：验证 bridge split package 和 high-level wrapper surface。
- split collection consumer：验证 split collection 包能被消费端 restore/build/native-copy。
- full package consumer：验证 full runtime package 的消费端路径。

当前 full package consumer 为：

```text
full package consumer: ready blocked-by-cuda-driver
full package consumer evidence scope: full-runtime-package-consumer-smoke-driver-blocked
full package consumer evidence classification: runtime-smoke-driver-blocked
full package consumer runtime-execution: False
full package consumer dependency-probe-only: True
full package consumer real-callback-proof: False
```

这表示 full package consumer 本身可用，smoke 已经请求，但执行被 CUDA driver/runtime compatibility 阻塞。

## Package consumer evidence schema

readiness summary 会把 package consumer evidence 拆成五个字段，避免把同一个 `ready blocked-by-cuda-driver` 误读为不同层次的 proof：

| 字段 | 当前值 | 读法 |
| --- | --- | --- |
| `packageConsumerEvidenceKind` | `full-runtime-package-consumer-smoke-driver-blocked` | full runtime package consumer 已经进入 smoke 路径，但最终是 driver-blocked evidence。 |
| `runtimeSmokeClassification` | `runtime-smoke-driver-blocked` | runtime smoke 被归类为 driver/runtime compatibility 阻塞。 |
| `isRuntimeExecutionEvidence` | `False` | 当前不是可晋级的 runtime execution proof。 |
| `isDependencyProbeOnly` | `True` | 当前只能作为 dependency probe/native-load/环境阻塞诊断。 |
| `isRealCallbackRuntimeProof` | `False` | 当前不是 TensorRT callback runtime proof。 |

因此，`Overall=ready` 可以和 `isRuntimeExecutionEvidence=False` 同时成立：前者是 package/readiness 层 ready，后者说明当前 smoke evidence 仍不能证明真实 runtime 执行完成。

## Runtime execution smoke

`runtime execution smoke` 是 runtime 执行层证据。当前状态：

```text
runtime execution smoke: blocked-by-cuda-driver
```

含义是 package consumer 程序已经实际启动 packaged runtime，并走到 CUDA runtime 边界，但 `cudaRuntimeGetVersion` 返回 CUDA error 35。它不是 API 缺失，也不是 package layout failure。

如果未来在兼容驱动上通过普通 smoke，也仍然不能自动证明 callback runtime。普通 smoke 通过只说明消费端程序运行成功；callback proof 需要单独的 `real-callback-runtime` markers。

## Callback evidence

当前报告中最需要谨慎阅读的是：

```text
real callback runtime evidence schema: schema-ready
real callback runtime evidence: blocked-by-cuda-driver; evidence-kind=not-present; proof=False
```

`schema-ready` 表示 proof 格式和审计规则已经写清楚。`proof=False` 表示真实 TensorRT callback runtime proof 没有完成。

以下都不是 proof：

- bridge-only dependency probe。
- compile-only package consumer。
- wrapper surface compiled。
- dry-run。
- copied-state。
- internal-runtime-prototype。
- safety-gate。
- design-gate。
- precheck。
- `blocked-by-cuda-driver`。

## Vendor blockers

报告底部会列出 vendor root：

```text
TensorRT=True
CUDA=True
cuDNN=True
Missing expected assets=0
vendor blockers: none
```

这说明本地 TensorRT/CUDA/cuDNN 文件根目录中的预期 DLL/LIB 都存在。它不保证当前 GPU driver 能运行某个 CUDA runtime 版本。

## readiness blockers

`readiness blockers: 0` 表示当前 readiness 脚本没有发现 package、consumer report 或 vendor input 层面的阻塞项。

不要把它解读成：

- 所有 deferred API 都已经提升为真实实现。
- 所有 samples 都在当前机器可运行。
- callback runtime proof 已完成。
- NVIDIA 二进制再分发许可已复核。

它只证明当前 runtime key 的 package readiness 条件满足。

## 推荐排查顺序

如果 readiness 不为 ready，按这个顺序看：

1. managed package 是否存在。
2. bridge package 是否存在。
3. bridge consumer report 是否存在并 ready。
4. split components 是否完整。
5. split collection package/consumer 是否 ready。
6. vendor roots 是否缺 DLL/LIB。
7. full runtime package 是否存在。
8. full package consumer 是否存在并 ready。
9. runtime smoke 是 `passed`、`not-requested`、`blocked-by-cuda-driver` 还是其它状态。

这样可以避免把 package 缺失、驱动不兼容、应用控制策略和真实 API 缺口混在一起。
