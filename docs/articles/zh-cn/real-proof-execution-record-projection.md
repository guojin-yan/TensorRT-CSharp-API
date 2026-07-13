# Real Proof Execution Record Projection

`real-proof-execution-record-projection` 是真实 proof 执行记录的形状投影层。它读取 `real-proof-runner-input-backfill` 的 6 条 track，将 Owner 待填字段投影成统一的 execution record schema，方便后续回填真实 host metadata、commands、logs、hashes 和 validator output。

它不是 proof，不执行 package publish，不代表 post-publish verification，也不能关闭 release issue。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofExecutionRecordProjection.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofExecutionRecordProjection.ps1 -Strict
```

## 输出

- `artifacts/final-release/real-proof-execution-record-projection.json`
- `artifacts/final-release/real-proof-execution-record-projection.md`
- `artifacts/final-release/real-proof-execution-record-projection-validation.json`
- `artifacts/final-release/real-proof-execution-record-projection-validation.md`

## 默认状态

- `recordKind=real-proof-execution-record-projection`
- `projectionState=blocked-real-proof-execution-record-input-required`
- `recordCount=6`
- `blockedRecordCount=6`
- `readyRecordCount=0`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## 记录字段

每条 record 都包含：

- `recordId` / `sourceTrackId` / `proofKind`
- `hostMetadata`
- `execution.commands`
- `logs`
- `hashes`
- `validatorOutputs`
- `forbiddenSubstituteChecks`
- `promotionFlags`
- `boundary`

## 边界

- placeholder 和 action-required 不是成功。
- hash match 只是完整性信号，不是 runtime proof。
- 本地 feed、`ProjectReference`、direct `.nupkg`、`DependencyProbe`、build-only、template、hash-only audit 都不能替代真实 proof。
- Windows handoff 不能替代 Linux runner proof。
- 只有 Owner 后续填入真实执行结果、日志、hash、validator output，并通过严格 validator 后，后续阶段才能考虑投影到可晋级 proof candidate。
