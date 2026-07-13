# Release Issue Final Close Decision

`release-issue-final-close-decision` 是最后的 Owner 输入合同。它要求 Owner 在真实 post-publish proof、final evidence freeze、release evidence bundle、release close candidate validation 和 rollback plan 都可审计后，明确给出最终关闭决定。

模板不是 close proof，不能自动关闭 issue，也不能替代 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueFinalCloseDecisionTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict
```

输出：

- `artifacts/final-release/release-issue-final-close-decision.template.json`
- `artifacts/final-release/release-issue-final-close-decision.template.md`
- `artifacts/final-release/release-issue-final-close-decision-validation.json`
- `artifacts/final-release/release-issue-final-close-decision-validation.md`

## 必须回填的真实确认

- 真实 post-publish proof 来自公开包源。
- clean consumer 位于仓库外。
- 没有 ProjectReference、local feed 或 direct `.nupkg`。
- runtime smoke 真实执行且 exit code 为 0。
- restore/build/smoke 日志和 SHA256 已审阅。
- rollback plan、rollback owner 和 rollback trigger 已审阅。
- Owner final close decision 明确为 `approved-to-close-after-real-proof`。

## 当前状态

模板状态应保持 `blocked-owner-final-close-decision-required`。这表示合同已经存在，但 Owner 真实外部 proof 和最终关闭决定尚未回填。只有真实 proof 与 strict close validator 全部通过后，release issue 才能进入人工关闭流程。
