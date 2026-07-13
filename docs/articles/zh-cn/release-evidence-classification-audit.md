# Release Evidence Classification Audit

`release-evidence-classification-audit` 是发布前的证据分类质量门，用来防止 `release-evidence-bundle` 中的非 proof 材料被误晋级为 runtime proof、owner approval、post-publish verification 或 release issue close approval。

它只审计分类边界，不执行 `dotnet nuget push`，不上传包，不批准公开发布，不关闭 issue，也不允许删除 deferred 记录。

## 审计范围

脚本会读取：

- `artifacts/final-release/release-evidence-bundle.json`

并检查：

- bundle 顶层必须保持 `canPublishPublicly=false`
- bundle 顶层必须保持 `canCloseReleaseIssue=false`
- bundle 顶层必须保持 `isRuntimeExecutionProof=false`
- design gate / precheck / dependency diagnostics 项必须保持 non-proof
- template / draft / runbook / owner guidance / input package / scaffold / local feed 项必须保持 non-proof
- `nonSubstituteProofKinds` 必须继续列出不可替代 proof 类型

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict
```

输出：

- `artifacts/final-release/release-evidence-classification-audit.json`
- `artifacts/final-release/release-evidence-classification-audit.md`

## 默认通过状态

当前阶段的期望状态是：

- `auditState=classification-audit-passed-non-proof-boundaries-intact`
- `auditPassed=true`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`

这表示分类边界保持正确，不表示项目已经具备发布 runtime proof。

## 仍然阻断发布的真实缺口

即使该审计通过，发布关闭仍然需要真实、可验证的：

- owner authorization
- clean package-consumer runtime smoke
- Linux runner proof
- real-model runtime proof
- post-publish verification
- release issue close record validation

在这些真实 proof 缺失前，`release-evidence-bundle` 必须继续保持 `blocked-evidence-incomplete`，并保持 `canPublishPublicly=false` / `canCloseReleaseIssue=false`。
