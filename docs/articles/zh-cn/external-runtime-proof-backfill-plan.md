# External Runtime Proof Backfill Plan

`external-runtime-proof-backfill-plan` 是兼容 CUDA/TensorRT 主机回填真实 external runtime proof 的阶段计划。它只输出 owner 可执行的步骤、证据字段和校验命令，不执行发布，也不能替代真实 smoke passed 记录。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofBackfillPlan.ps1
```

输出：

- `artifacts/final-release/external-runtime-proof-backfill-plan.json`
- `artifacts/final-release/external-runtime-proof-backfill-plan.md`

默认状态必须保持：

- `recordKind=external-runtime-proof-backfill-plan`
- `planState=blocked-compatible-host-proof-required`
- `performsPublish=false`
- `approvesPublicRelease=false`
- `canPromoteRuntimeProof=false`
- `canCloseReleaseIssue=false`
- `runtimeProofStatus=blocked-by-cuda-driver`
- `proofClassification=template-only`

## 回填顺序

计划固定把真实 proof 回填拆成七步：

1. 刷新 `external-runtime-proof-record.input-template.json`。
2. 刷新 owner handoff、compatible-host runbook 和 collection bundle。
3. 在兼容 CUDA/TensorRT 主机运行 package consumer smoke。
4. 捕获 consumed managed/runtime nupkg 与 smoke log 的 SHA256。
5. 把 input-template 回填成真实 `external-runtime-proof-record.json`。
6. 使用 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 验证真实记录。
7. 仅在 validator 晋级后刷新 release evidence、owner input、owner decision 和 freeze summary。

## 证据边界

- backfill plan 不是 runtime proof。
- handoff、runbook、collection bundle、draft、example 和 template 都不是 promotable proof。
- `blocked-by-cuda-driver` 不是 smoke passed。
- `DependencyProbe BridgeInitialized` 不是 runtime execution proof。
- final package review 只是 local package inventory，不是 public package proof。
- 只有真实 `external-runtime-proof-record.json` 在兼容主机上通过 `-RequireExistingLog -FailOnNotProof`，才能把 `canPromoteRuntimeProof` 晋级。

## 必填回填项

真实记录至少需要：

- clean consumer project name/path，且无 `ProjectReference`。
- managed/runtime nupkg SHA256。
- host OS、GPU、driver、CUDA runtime、TensorRT runtime/line、cuDNN version。
- restore/build/smoke command，其中 smoke command 包含 `--runtime-package-key`。
- smoke `exitCode=0`、`smokeStatus=passed`、native assets copied。
- smoke log path 与 SHA256，且 validator 重新计算匹配。
- stdoutSummary 与 stderrSummary；stderr 为空时写 `no-stderr-emitted`。

## 与发布链路的关系

owner command plan 会聚合本计划的 `planState` 和 step count，让 release owner 看到 external proof 的剩余工作。但该聚合不改变 `performsPublish=false`，也不会把 `canMaterializeExecutableCommands` 改成 true。
