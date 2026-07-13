# Final Owner StrictClose Execution Order

`final-owner-strict-close-execution-order` 是最终 Owner 执行顺序归档，用来把已经分散在 action worklist、execution package、clean consumer runbook、post-publish runbook、public publish lane、command cross-check、readiness checkpoint、close blocker dashboard 和 Owner 输入合同里的人工步骤收敛成一条可执行但仍然 blocked 的顺序。

它不是 runtime proof，不是 post-publish proof，不是 publish approval，不是 release close approval，也不是 package push。它不会执行 `dotnet nuget push`，不会导入真实 Owner stdout/stderr/log/hash，也不会把本地 dry-run、candidate、dashboard、ProjectReference、direct `.nupkg`、local feed 或 build-only 结果提升为 proof。

## 覆盖范围

- 7 个 final owner proof action worklist action。
- 7 个 final owner execution package step。
- 9 个 clean external package consumer owner runbook step。
- 6 个 post-publish owner verification runbook step。
- 10 个 public publish final owner execution lane。
- 11 个 public publish command cross-check item。
- 12 个 final owner close readiness check。
- 19 个 final release close blocker。
- 5 个 Owner input contract surface。
- 14 个 Owner input canonical field。
- 2 个 Owner runbook input。

## 执行边界

该归档只提供 Owner 手动执行顺序和输入/输出/验证器对照。每个步骤都必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

Owner 仍需在真实兼容主机和真实公开包通道上执行命令，并导入真实 stdout、stderr、日志路径、hash、host/package metadata、rollback review 和最终 close decision。只有这些真实输入通过 strict validator 后，后续阶段才允许判断是否能进入 release close。

## 生成与验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerInputContractConvergence.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerInputContractConvergence.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerStrictCloseExecutionOrder.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerStrictCloseExecutionOrder.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict
```

输出文件：

- `artifacts/final-release/final-owner-strict-close-execution-order.json`
- `artifacts/final-release/final-owner-strict-close-execution-order.md`
- `artifacts/final-release/final-owner-strict-close-execution-order-validation.json`
- `artifacts/final-release/final-owner-strict-close-execution-order-validation.md`

## 与发布证据链的关系

Release evidence bundle 会把该归档作为 `final-owner-strict-close-execution-order` evidence item 收录，但 `passed=false`，并在 `sourceArtifacts` 中记录 JSON、Markdown 与 validation 输出。Release evidence classification audit 必须继续把它识别为 non-proof 项，防止最终 Owner 指南被误用为真实运行证明、发布批准或 release close 批准。

当前项目公开示例命名以 `samples/YoloVision` 为目标，不恢复旧 YOLO 检测样例名作为公开入口名称。后续真实 Owner 执行应继续围绕可发布项目边界、clean package consumer、`applications/TensorRtExec`、`samples/YoloVision` 和 StrictClose validator 做证据闭环。
