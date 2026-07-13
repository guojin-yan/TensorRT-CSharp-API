# Owner Input Cross-Hash Audit

`owner-input-cross-hash-audit` 是 Owner 输入链路的本地 artifact/path/hash 一致性审计。它把 release evidence bundle、final evidence freeze、post-publish validation、release close candidate validation、final close decision validation、real external proof overlay、release issue close record overlay candidate 和 owner external execution result backfill kit 的本地路径与 SHA256 串起来，检查这些本地产物是否互相引用一致。

它不是外部 proof，不会执行发布，不会批准公开发布，也不会关闭 release issue。当前状态必须保持：

- `auditState=blocked-owner-input-cross-hash-audit-owner-proof-required`
- `validationState=blocked-owner-input-cross-hash-audit-owner-proof-required`
- `mismatchedHashCount=0`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerInputCrossHashAudit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerInputCrossHashAudit.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/owner-input-cross-hash-audit.json`
- `artifacts/final-release/owner-input-cross-hash-audit.md`
- `artifacts/final-release/owner-input-cross-hash-audit-validation.json`
- `artifacts/final-release/owner-input-cross-hash-audit-validation.md`

## 审计范围

cross-hash audit 只检查本地 evidence chain 的可追溯性：

- artifact path 是否存在。
- recorded SHA256 是否与当前文件一致。
- overlay candidate、final freeze、post-publish validation 和 close decision validation 是否引用同一批本地证据。
- owner external execution result backfill kit validation 是否仍保持 blocked/non-proof。

## 不能误读

- `mismatchedHashCount=0` 只表示本地 hash 一致，不表示 runtime proof 已通过。
- 本地 hash 一致不能替代仓库外 clean consumer smoke、真实发布渠道、host metadata、runtime log 和 post-publish verification。
- placeholder、missing log、missing SHA256 或 host metadata 缺失时必须继续 blocked。
- 即使所有 hash 都匹配，也不能把 `canCloseReleaseIssue` 提升为 `true`。
