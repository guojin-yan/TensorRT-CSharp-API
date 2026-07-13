# Real Owner Evidence StrictValidator Orchestration

`real-owner-evidence-strict-validator-orchestration` 是真实 Owner 证据导入与 StrictValidator 联调层。它把 Owner 输入合同、最终 Owner 执行顺序、public publish 输入、clean consumer 输入、post-publish proof record 合同、owner external proof import、real proof candidate、strict record、public package hash cross-check、final close record validator 和 final publish gate 放到同一张可审计矩阵里。

它仍然是 blocked/non-proof coordination map，不是 runtime proof，不是 post-publish proof，不是 publish approval，不是 release close approval，也不是 package push。`failedBlockerCount=0` 只表示结构验证没有 blocker，不能解释成 proof ready。

## 覆盖范围

- 11 个上游 source record：Owner 输入合同、StrictClose 执行顺序、public publish owner input、package consumer owner input、post-publish contract、owner import、candidate、strict record、hash gate、close validator、final publish gate。
- 16 个字段级 readiness row：覆盖 public package URL/hash、package id/version/source、clean consumer 项目与日志、post-publish install/run 日志、stdout/stderr、host metadata、non-substitute confirmation、rollback review、final close decision。
- 6 个以上 strict validator consumer：覆盖 hash cross-check、owner import、strict record、close validator、final publish gate、classification audit 等边界。

## 字段边界

每个字段都保持 `blocked-real-owner-input-required`，并明确以下替代物不能满足要求：

- local dry-run
- ProjectReference
- direct `.nupkg`
- local feed
- candidate
- dashboard
- runbook
- build-only
- template 或 draft

这些字段必须来自真实 Owner 外部执行证据，并由 strict validator 消费。当前联调层只负责把字段、输入面、validator 和 blockedUntil 串起来，不导入真实证据，也不提升 proof。

## 生成与验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerInputContractConvergence.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerInputContractConvergence.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerStrictCloseExecutionOrder.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerStrictCloseExecutionOrder.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealOwnerEvidenceStrictValidatorOrchestration.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealOwnerEvidenceStrictValidatorOrchestration.ps1 -Strict
```

输出文件：

- `artifacts/final-release/real-owner-evidence-strict-validator-orchestration.json`
- `artifacts/final-release/real-owner-evidence-strict-validator-orchestration.md`
- `artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.json`
- `artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.md`

## 发布证据关系

Release Evidence Bundle 会把该联调层作为 `real-owner-evidence-strict-validator-orchestration` evidence item 收录，且必须保持 `passed=false`。Release Evidence Classification Audit 必须继续把它识别为 non-proof，防止字段矩阵、联调图或 0 个结构 blocker 被误用为真实运行证明、发布批准或 release close 批准。

当前公开 YOLO 示例入口保持 `samples/YoloVision`。真实 Owner 证据后续应继续围绕 public package、clean external consumer、post-publish clean consumer、`applications/TensorRtExec` 和 StrictClose validator 做闭环。
